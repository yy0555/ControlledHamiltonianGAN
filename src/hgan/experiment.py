import logging
import os
import os.path
import time
import glob
import numpy as np
import skvideo.io
from skimage.transform import resize
import importlib
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable

import hgan.data
from hgan.configuration import save_config
from hgan.models import GRU, HNNSimple, HNNPhaseSpace, HNNMass
from hgan.dataset import (
    RealtimeDataset,
    HGNRealtimeDataset,
    ToyPhysicsDatasetNPZ,
    RealPendulumVideoDataset,
)
from hgan.utils import setup_reproducibility, timeSince
from hgan.fvd import compute_fvd
from hgan.models import Discriminator_I, Discriminator_V, Generator_I
from hgan.updates import update_models


logger = logging.getLogger(__name__)


def _unwrap(module):
    """Return the underlying module if wrapped in DDP, else the module itself."""
    return module.module if isinstance(module, DDP) else module


class Experiment:
    def __init__(self, config):
        self.dataloader = None
        self.model_names = (
            "Di",
            "Dv",
            "Gi",
            "rnn",
            "optim_Di",
            "optim_Dv",
            "optim_Gi",
            "optim_rnn",
            "system_embedding",
        )
        self.Di = self.Dv = self.Gi = self.rnn = None
        self.optim_Di = self.optim_Dv = self.optim_Gi = self.optim_rnn = None
        self.config = config

        self._init_derived_attributes(self.config)
        self._init_dataloader(self.config)
        self._init_models(self.config)

    def _init_derived_attributes(self, config):

        # Make all keys available in config.experiment as attributes in this object, for convenience
        for k, v in config.experiment.items():
            setattr(self, k, v)

        # Only 1 GPU supported for now
        self.ngpu = 1

        if config.experiment.system_name is not None and config.paths.input is not None:
            self.datapath = os.path.join(
                config.paths.input, config.experiment.system_name
            )
        else:
            self.datapath = None

        # DDP detection: torchrun sets LOCAL_RANK/RANK/WORLD_SIZE. If those
        # env vars are present and WORLD_SIZE>1 we run in distributed mode.
        local_rank_env = os.environ.get("LOCAL_RANK", "")
        world_size_env = os.environ.get("WORLD_SIZE", "1")
        self.world_size = int(world_size_env) if world_size_env else 1
        self.local_rank = int(local_rank_env) if local_rank_env != "" else -1
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_ddp = self.world_size > 1 and self.local_rank >= 0
        self.is_main = self.rank == 0  # true for single-proc runs as well

        if self.is_ddp:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        elif config.experiment.gpu is None or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = f"cuda:{config.experiment.gpu}"

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.label = torch.FloatTensor().to(self.device)

        self.ndim_p = self.ndim_q = int(self.ndim_epsilon / 2)
        self.betas = tuple(float(b) for b in config.experiment.betas.split(","))

        self.ndim_q2 = int(self.ndim_q**2)

        if config.experiment.architecture == "hnn_mass":
            self.nz = (
                self.ndim_q + int((self.ndim_q2 + self.ndim_q) / 2) + self.ndim_content
            )
        else:
            self.nz = self.ndim_content + self.ndim_epsilon

    def _init_dataloader(self, config):
        if config.experiment.rt_data_generator == "hgn":
            dataset = HGNRealtimeDataset(
                ndim_label=config.experiment.ndim_label,
                ndim_physics=config.experiment.ndim_physics,
                ndim_color=config.experiment.ndim_color,
                system_name=config.experiment.system_name,
                num_frames=config.video.generator_frames,
                delta=0.05,
                train=True,
                system_physics_constant=config.experiment.system_physics_constant,
                system_color_constant=config.experiment.system_color_constant,
                system_friction=config.experiment.system_friction,
                total_frames=config.video.real_total_frames,
                img_size=config.experiment.img_size,
                normalize=config.video.normalize,
            )
        elif config.experiment.rt_data_generator == "dm":
            dataset = RealtimeDataset(
                ndim_physics=config.experiment.ndim_physics,
                system_name=config.experiment.system_name,
                num_frames=config.video.generator_frames,
                delta=0.05,
                train=True,
                system_physics_constant=config.experiment.system_physics_constant,
                system_color_constant=config.experiment.system_color_constant,
                system_friction=config.experiment.system_friction,
                total_frames=config.video.real_total_frames,
                img_size=config.experiment.img_size,
                normalize=config.video.normalize,
            )
        elif config.experiment.rt_data_generator == "real_pendulum":
            data_dir = config.experiment.real_pendulum_data_dir
            video_filename = getattr(
                config.experiment, "real_pendulum_video_filename", "DP_free_drop_video.mp4"
            )
            stride = getattr(config.experiment, "real_pendulum_stride", 1)
            motion_percentile = getattr(
                config.experiment, "real_pendulum_motion_percentile", 0
            )
            dataset = RealPendulumVideoDataset(
                data_dir=data_dir,
                video_filename=video_filename,
                num_frames=config.video.generator_frames,
                img_size=config.experiment.img_size,
                ndim_channel=config.experiment.ndim_channel,
                ndim_label=config.experiment.ndim_label,
                ndim_physics=config.experiment.ndim_physics,
                ndim_color=config.experiment.ndim_color,
                stride=stride,
                normalize=config.video.normalize,
                train=True,
                motion_percentile=float(motion_percentile),
            )
        else:
            dataset = ToyPhysicsDatasetNPZ(
                datapath=self.datapath, num_frames=config.video.generator_frames
            )

        if len(dataset) == 0:
            raise RuntimeError("No videos found!")

        if self.is_ddp:
            self._sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.dataloader = DataLoader(
                dataset,
                batch_size=config.experiment.batch_size,
                sampler=self._sampler,
                pin_memory=True,
                drop_last=True,
            )
        else:
            self._sampler = None
            self.dataloader = DataLoader(
                dataset,
                batch_size=config.experiment.batch_size,
                shuffle=True,
                pin_memory=True,
            )

    def _init_models(self, config):
        n_label_and_props = (
            config.experiment.ndim_label
            + config.experiment.ndim_physics
            + config.experiment.ndim_color
        )
        self.Di = Discriminator_I(
            self.ndim_channel,
            self.ndim_discriminator_filter,
            ngpu=self.ngpu,
            n_label_and_props=n_label_and_props,
        ).to(self.device)
        self.Dv = Discriminator_V(
            self.ndim_channel,
            self.ndim_discriminator_filter,
            T=config.video.discriminator_frames,
            n_label_and_props=n_label_and_props,
        ).to(self.device)
        self.Gi = Generator_I(
            self.ndim_channel,
            self.ndim_generator_filter,
            self.nz + self.ndim_label + self.ndim_color,
            ngpu=self.ngpu,
        ).to(self.device)

        rnn_class = {
            "gru": GRU,
            "hnn_simple": HNNSimple,
            "hnn_phase_space": HNNPhaseSpace,
            "hnn_mass": HNNMass,
        }[config.experiment.architecture]

        if config.experiment.architecture in ("hnn_phase_space", "hnn_mass"):
            self.rnn = rnn_class(
                device=self.device,
                input_size=self.ndim_epsilon + self.ndim_label + self.ndim_physics,
                hidden_size=self.hidden_size,
                output_size=self.ndim_epsilon,
                ndim_physics=self.ndim_physics,
                ndim_label=self.ndim_label,
            ).to(self.device)
        else:
            self.rnn = rnn_class(
                device=self.device,
                input_size=self.ndim_epsilon,
                hidden_size=self.hidden_size,
            ).to(self.device)

        self.rnn.initWeight()

        # Wrap models in DDP once, *before* creating optimizers so the
        # optimizer sees the DDP-wrapped params. `find_unused_parameters=True`
        # is required because Discriminator_I/V's `label_handler` has an
        # unused bias when n_label_and_props == 0.
        if self.is_ddp:
            self.Di = DDP(
                self.Di, device_ids=[self.local_rank], find_unused_parameters=True
            )
            self.Dv = DDP(
                self.Dv, device_ids=[self.local_rank], find_unused_parameters=True
            )
            self.Gi = DDP(
                self.Gi, device_ids=[self.local_rank], find_unused_parameters=True
            )
            self.rnn = DDP(
                self.rnn, device_ids=[self.local_rank], find_unused_parameters=True
            )

        self.optim_Di = torch.optim.Adam(
            self.Di.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_Dv = torch.optim.Adam(
            self.Dv.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_Gi = torch.optim.Adam(
            self.Gi.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_rnn = torch.optim.Adam(
            self.rnn.parameters(), lr=self.learning_rate, betas=self.betas
        )

    @property
    def system_embedding(self):
        return self.dataloader.dataset.system_embedding

    def saved_epochs(self):
        saved_pths = sorted(glob.glob(self.config.paths.output + "/Di_*.pth"))
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in saved_pths]
        epochs = [int(filename.split("_")[-1]) for filename in filenames]
        return epochs

    def load_epoch(self, epoch=None, device=None):
        if epoch is None:
            saved_epochs = self.saved_epochs()
            if not saved_epochs:
                return 0
            epoch = saved_epochs[-1]

        for which in self.model_names:
            file_path = os.path.join(
                self.config.paths.output, f"{which}_{epoch:0>6}.pth"
            )
            model = getattr(self, which)
            state = torch.load(file_path, map_location=device)
            _unwrap(model).load_state_dict(state) if isinstance(
                model, torch.nn.Module
            ) else model.load_state_dict(state)

        return epoch

    def eval(self):
        for which in self.model_names:
            if not which.startswith("optim"):
                model = getattr(self, which)
                model.eval()

    def no_eval(self):
        for which in self.model_names:
            if not which.startswith("optim"):
                model = getattr(self, which)
                model.train()

    def save_video(self, folder, video, epoch=None, filename=None, prefix="video_"):
        os.makedirs(folder, exist_ok=True)
        video = np.asarray(video, dtype=np.float32)
        # Generator ends in Tanh → output in [-1, 1] when normalize=1, else
        # already in [0, 1]. Map both back to [0, 1] before uint8 conversion,
        # then clip so Tanh excursions past ±1 don't wrap around in uint8 cast.
        if self.config.video.normalize:
            video = (video + 1.0) * 0.5
        video = np.clip(video, 0.0, 1.0)
        outputdata = (video * 255.0).astype(np.uint8)
        # libx264 expects 3-channel frames. If the generator runs in grayscale
        # (ndim_channel=1) the last axis is size 1 — replicate to RGB.
        if outputdata.ndim == 4 and outputdata.shape[-1] == 1:
            outputdata = np.repeat(outputdata, 3, axis=-1)
        elif outputdata.ndim == 3:
            outputdata = np.stack([outputdata] * 3, axis=-1)
        filename = filename or f"{prefix}{epoch:0>6}"
        file_path = os.path.join(folder, f"{filename}.mp4")
        # imageio (via imageio_ffmpeg) ships its own ffmpeg binary, so no
        # system ffmpeg/ffprobe is required. This matches skvideo's contract.
        import imageio.v3 as iio
        iio.imwrite(file_path, outputdata, fps=30, plugin="FFMPEG", codec="libx264")

    def save_epoch(self, epoch):
        for which in self.model_names:
            file_path = os.path.join(
                self.config.paths.output, f"{which}_{epoch:0>6}.pth"
            )
            model = getattr(self, which)
            # Strip DDP wrapper so checkpoints load cleanly in non-DDP inference.
            state = (
                _unwrap(model).state_dict()
                if isinstance(model, torch.nn.Module)
                else model.state_dict()
            )
            torch.save(state, file_path)

    def get_random_content_vector(self, batch_size, d_C, device, n_frames):
        z_C = Variable(torch.randn(batch_size, d_C))
        #  repeat z_C to (batch_size, n_frames, d_C)
        z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
        z_C = z_C.to(device)

        return z_C

    def compute_phase_space_motion_vector(
        self, batch_size, d_E, d_L, d_P, device, dataset, n_frames, rnn, label_and_props
    ):
        eps_motion = Variable(torch.randn(batch_size, d_E))
        eps_motion = eps_motion.to(device)
        if label_and_props is not None:
            eps = torch.cat([label_and_props, eps_motion], dim=1)
        else:
            eps = eps_motion
        _unwrap(rnn).initHidden(batch_size)
        # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
        z_M, dz_M = rnn(eps, n_frames)
        z_M = z_M.transpose(1, 0)
        if dz_M is not None:
            dz_M = dz_M.transpose(1, 0)
        return z_M, dz_M, eps_motion

    def get_phase_space_sample(
        self,
        batch_size,
        d_C,
        d_E,
        d_L,
        d_P,
        device,
        nz,
        dataset,
        rnn,
        n_frames,
        label_and_props=None,
    ):
        z_C = self.get_random_content_vector(
            batch_size, d_C, device, n_frames
        )  # (batch_size, n_frames, ndim_content)
        z_M, dz_M, eps_motion = self.compute_phase_space_motion_vector(
            batch_size=batch_size,
            d_E=d_E,
            d_L=d_L,
            d_P=d_P,
            device=device,
            dataset=dataset,
            n_frames=n_frames,
            rnn=rnn,
            label_and_props=label_and_props,
        )
        z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

        return z.view(batch_size, n_frames, nz, 1, 1), dz_M, eps_motion

    def compute_simple_motion_vector(self, batch_size, d_E, device, n_frames, rnn):
        eps = Variable(torch.randn(batch_size, d_E))
        eps = eps.to(device)

        _unwrap(rnn).initHidden(batch_size)
        # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
        z_M = rnn(eps, n_frames).transpose(1, 0)
        return z_M

    def compute_mass_motion_vector(self, batch_size, d_E, d_N, device, n_frames, rnn):
        eps = Variable(torch.randn(batch_size, d_E))
        Z_mass = Variable(torch.randn(batch_size, d_N))

        eps = eps.to(device)
        Z_mass = Z_mass.to(device)

        _unwrap(rnn).initHidden(batch_size)
        # notice that 1st dim of hnn outputs is seq_len, 2nd is batch_size
        z_M, z_mass = rnn(eps, Z_mass, n_frames)

        z_M = z_M.transpose(1, 0)
        z_mass = z_mass.unsqueeze(1).repeat(1, n_frames, 1)

        return z_M, z_mass

    def get_simple_sample(self, batch_size, d_C, d_E, nz, device, rnn, n_frames):
        z_C = self.get_random_content_vector(batch_size, d_C, device, n_frames)
        z_M = self.compute_simple_motion_vector(batch_size, d_E, device, n_frames, rnn)
        z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

        return z.view(batch_size, n_frames, nz, 1, 1), None

    def get_mass_sample(self, batch_size, d_C, d_E, d_N, device, nz, rnn, n_frames):
        z_C = self.get_random_content_vector(batch_size, d_C, device, n_frames)
        z_M, z_mass = self.compute_mass_motion_vector(
            batch_size, d_E, d_N, device, n_frames, rnn
        )
        z = torch.cat((z_M, z_mass, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
        return z.view(batch_size, n_frames, nz, 1, 1), None

    def get_latent_sample(self, batch_size, n_frames, label_and_props=None):
        # TODO: Do this through inheritance
        if self.architecture in ("gru", "hnn_simple"):
            return self.get_simple_sample(
                batch_size,
                self.ndim_content,
                self.ndim_epsilon,
                self.nz,
                self.device,
                self.rnn,
                n_frames,
            )
        elif self.architecture in ("hnn_phase_space",):
            return self.get_phase_space_sample(
                batch_size=batch_size,
                d_C=self.ndim_content,
                d_E=self.ndim_epsilon,
                d_L=self.ndim_label,
                d_P=self.ndim_physics,
                device=self.device,
                nz=self.nz,
                dataset=self.dataloader.dataset,
                rnn=self.rnn,
                n_frames=n_frames,
                label_and_props=label_and_props,
            )
        else:
            return self.get_mass_sample(
                batch_size=batch_size,
                d_C=self.ndim_content,
                d_E=self.ndim_epsilon,
                d_N=self.ndim_q2,
                device=self.device,
                nz=self.nz,
                rnn=self.rnn,
                n_frames=n_frames,
            )

    def trim_video(self, video, n_frame):
        # Trim a (batch_size, T, ...) video to (batch_size, n_frame, ...)
        start = np.random.randint(0, video.size(1) - n_frame + 1)
        end = start + n_frame
        return video[:, start:end, ...]

    def get_fake_data(self, n_frames=None, label_and_props=None, colors=None):
        n_frames = n_frames or self.config.video.generator_frames
        # Z.size() => (batch_size, n_frames, nz, 1, 1)
        Z, dz, _ = self.get_latent_sample(
            batch_size=self.batch_size,
            n_frames=n_frames,
            label_and_props=label_and_props,
        )
        # trim => (batch_size, T, nz, 1, 1)
        Z = self.trim_video(video=Z, n_frame=n_frames)
        Z_reshape = Z.contiguous().view(self.batch_size * n_frames, self.nz, 1, 1)

        # Append label+color information; duplicating it for each frame
        # (batch_size, n) => (batch_size * n_frames, n, 1, 1)

        label = label_and_props[:, : self.ndim_label]
        label_and_colors = torch.cat((label, colors), dim=1)
        label_and_colors_reshape = (
            label_and_colors.unsqueeze(1)
            .repeat(1, n_frames, 1)
            .contiguous()
            .view(self.batch_size * n_frames, -1, 1, 1)
        )
        Z_reshape = torch.cat((Z_reshape, label_and_colors_reshape), dim=1)

        fake_videos = self.Gi(Z_reshape)

        fake_videos = fake_videos.view(
            self.batch_size, n_frames, self.ndim_channel, self.img_size, self.img_size
        )
        # transpose => (batch_size, nc, T, img_size, img_size)
        fake_videos = fake_videos.transpose(2, 1)
        # img sampling
        fake_img = fake_videos[:, :, np.random.randint(0, n_frames), :, :]

        fake_data = {"videos": fake_videos, "img": fake_img, "latent": Z, "dlatent": dz}

        return fake_data

    def save_fake_images(self, generated_img_path, n=1, video_length=50):
        for i in range(n):
            fake_data = self.get_fake_data()
            # (batch_size, T, nc, img_size, img_size)
            fake_data_np = (
                fake_data["videos"].permute(0, 2, 1, 3, 4).detach().cpu().numpy()
            )
            filename = generated_img_path + str(i).zfill(4)
            np.save(filename, fake_data_np)

    def get_real_data(self, device=None, dataloader=None):
        device = device or self.device
        dataloader = dataloader or self.dataloader
        label_and_props = torch.tensor([])
        colors = torch.tensor([])
        next_item = next(iter(dataloader))
        if isinstance(next_item, (tuple, list)):
            real_videos = next_item[0]
            if len(next_item) > 2:
                colors = next_item[2]
            if len(next_item) > 1:
                label_and_props = next_item[1]
        else:
            real_videos = next_item

        real_videos = real_videos.to(
            device
        )  # (batch_size, ndim_channels, n_frames, img_size, img_size)
        real_videos = Variable(real_videos)
        label_and_props = label_and_props.to(device)
        label_and_props = Variable(label_and_props)
        colors = colors.to(device)
        colors = Variable(colors)

        real_videos_frames = real_videos.shape[2]

        real_img = real_videos[:, :, np.random.randint(0, real_videos_frames), :, :]

        real_data = {
            "videos": real_videos,
            "img": real_img,
            "label_and_props": label_and_props,
            "colors": colors,
        }

        return real_data

    def fvd(
        self,
        real_videos=None,
        fake_videos=None,
        max_videos=None,
        device="cpu",
        label_and_props=None,
        colors=None,
    ):

        if real_videos is None:
            real_data = self.get_real_data()
            real_videos = real_data[
                "videos"
            ]  # (batch_size, n_channels, n_frames, height, width)
            label_and_props = real_data[
                "label_and_props"
            ]  # (batch_size, 1, ndim_label+ndim_physics)
            colors = real_data["colors"]  # (batch_size, 1, ndim_color)

        if fake_videos is None:
            fake_data = self.get_fake_data(
                label_and_props=label_and_props, colors=colors
            )
            fake_videos = fake_data[
                "videos"
            ]  # (batch_size, n_channels, n_frames, height, width)

        # Use shape (samples, n_frames, n_channels, height, width)
        real_videos = real_videos.detach().cpu().numpy().transpose(0, 2, 1, 3, 4)
        fake_videos = fake_videos.detach().cpu().numpy().transpose(0, 2, 1, 3, 4)

        with importlib.resources.path(hgan.data, "i3d_torchscript.pt") as i3d_path:
            detector = torch.jit.load(i3d_path).eval().to(device)

        batch_size, num_frames, num_channels, height, width = real_videos.shape
        # I3D expects 3-channel video. For grayscale training (ndim_channel=1)
        # we replicate the channel to feed the detector.
        if num_channels == 1:
            real_videos = np.repeat(real_videos, 3, axis=2)
            fake_videos = np.repeat(fake_videos, 3, axis=2)
            num_channels = 3
        assert num_channels == 3, "Inputs should be 3 channels"

        resized_real_videos = []
        for vid in real_videos[:max_videos]:
            resized_video = np.asarray([resize(img, (3, 224, 224)) for img in vid])
            resized_real_videos.append(resized_video)
        resized_real_videos = np.array(resized_real_videos)

        resized_fake_videos = []
        for vid in fake_videos[:max_videos]:
            resized_video = np.asarray([resize(img, (3, 224, 224)) for img in vid])
            resized_fake_videos.append(resized_video)
        resized_fake_videos = np.array(resized_fake_videos)

        # detector expects inputs of shape (batch_size, num_channels, num_frames, height, width)
        resized_real_videos = (
            torch.from_numpy(resized_real_videos).to(device).permute(0, 2, 1, 3, 4)
        )
        resized_fake_videos = (
            torch.from_numpy(resized_fake_videos).to(device).permute(0, 2, 1, 3, 4)
        )

        detector_kwargs = {
            "rescale": False,
            "resize": False,
            "return_features": True,  # Return raw features before the softmax layer.
        }
        feats_real = (
            detector(resized_real_videos, **detector_kwargs).detach().cpu().numpy()
        )
        feats_fake = (
            detector(resized_fake_videos, **detector_kwargs).detach().cpu().numpy()
        )

        fvd = compute_fvd(real_activations=feats_real, generated_activations=feats_fake)
        return fvd

    def train_step(self):
        real_data = self.get_real_data()
        label_and_props = real_data["label_and_props"]
        colors = real_data["colors"]

        fake_data = self.get_fake_data(label_and_props=label_and_props, colors=colors)

        err, mean = update_models(
            rnn_type=self.architecture,
            label=self.label,
            criterion=self.criterion,
            q_size=self.ndim_q,
            batch_size=self.batch_size,
            cyclic_coord_loss=self.cyclic_coord_loss,
            r1_gamma=self.r1_gamma,
            model_di=self.Di,
            model_dv=self.Dv,
            model_gi=self.Gi,
            model_rnn=self.rnn,
            optim_di=self.optim_Di,
            optim_dv=self.optim_Dv,
            optim_gi=self.optim_Gi,
            optim_rnn=self.optim_rnn,
            real_data=real_data,
            fake_data=fake_data,
            discriminator_gamma=self.discriminator_gamma,
            generator_gamma=self.generator_gamma,
        )

        return err, mean, real_data, fake_data

    def train(self):

        for which in self.model_names:
            if not which.startswith("optim"):
                model = getattr(self, which)
                model.eval()

        if self.is_main:
            save_config(self.config.paths.output)
        setup_reproducibility(seed=self.seed + self.rank)

        if self.retrain:
            start_epoch = 0
        else:
            start_epoch = self.load_epoch()

        start_time = time.time()
        for epoch in range(start_epoch + 1, self.n_epoch + 1):
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            err, mean, real_data, fake_data = self.train_step()

            real_videos = real_data["videos"]
            fake_videos = fake_data["videos"]

            last_epoch = epoch == self.n_epoch

            if epoch % self.calculate_fvd_every == 0 or last_epoch:
                fvd = self.fvd(real_videos=real_videos, fake_videos=fake_videos)
                if self.is_main:
                    logger.info(f"FVD = {fvd}")

            if self.is_main and (epoch % self.print_every == 0 or last_epoch):
                logger.info(
                    "[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f"
                    % (
                        epoch,
                        self.n_epoch,
                        timeSince(start_time),
                        err["Di"],
                        err["Dv"],
                        err["Gi"],
                        err["Gv"],
                        mean["Di_real"],
                        mean["Di_fake"],
                        mean["Dv_real"],
                        mean["Dv_fake"],
                    )
                )

            if self.is_main and (epoch % self.save_fake_video_every == 0 or last_epoch):
                self.save_video(
                    self.config.paths.output,
                    fake_videos[0].detach().cpu().numpy().transpose(1, 2, 3, 0),
                    epoch=epoch,
                    prefix="fake_",
                )

            if self.is_main and (epoch % self.save_real_video_every == 0 or last_epoch):
                self.save_video(
                    self.config.paths.output,
                    real_videos[0].detach().cpu().numpy().transpose(1, 2, 3, 0),
                    epoch=epoch,
                    prefix="real_",
                )

            if self.is_main and (epoch % self.save_model_every == 0 or last_epoch):
                self.save_epoch(epoch)

        if self.is_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


class ExperimentOld(Experiment):
    def saved_epochs(self):
        saved_pths = sorted(
            glob.glob(self.config.paths.output + "/Discriminator_I_*.model")
        )
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in saved_pths]
        epochs = [int(filename.split("-")[-1]) for filename in filenames]
        return epochs

    def load_epoch(self, epoch=None):
        if epoch is None:
            saved_epochs = self.saved_epochs()
            if not saved_epochs:
                return 0
            epoch = saved_epochs[-1]

        logger.info(f"Loading model from epoch {epoch}")
        names = {
            "Di": "Discriminator_I",
            "Dv": "Discriminator_V",
            "Gi": "Generator_I",
            "rnn": self.rnn.__class__.__name__,
            "optim_Di": "Discriminator_I",
            "optim_Dv": "Discriminator_V",
            "optim_Gi": "Generator_I",
            "optim_rnn": self.rnn.__class__.__name__,
        }
        exts = {
            "Di": "model",
            "Dv": "model",
            "Gi": "model",
            "rnn": "model",
            "optim_Di": "state",
            "optim_Dv": "state",
            "optim_Gi": "state",
            "optim_rnn": "state",
        }
        for which in self.model_names:
            file_path = os.path.join(
                self.config.paths.output, f"{names[which]}_epoch-{epoch}.{exts[which]}"
            )
            model = getattr(self, which)
            model.load_state_dict(torch.load(file_path))

        return epoch
