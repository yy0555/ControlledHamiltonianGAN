import os
import glob
import logging
import skvideo.io
from skimage.transform import resize
import numpy as np
import torch
from torch.utils.data import Dataset
import jax
import functools
from hgan.dm_hamiltonian_dynamics_suite import datasets
from hgan.configuration import config
from hgan.dm_datasets import all_systems, constant_physics, variable_physics
from hgan.hgn_datasets import (
    all_systems_hgn,
    constant_physics_hgn,
    variable_physics_hgn,
)
from hgan.hgn.environments.environment_factory import EnvFactory

logger = logging.getLogger(__name__)


class AviDataset(Dataset):
    def __init__(self, datapath, T):
        self.T = T
        self.datapath = os.path.join(datapath, "resized_data")
        self.files = glob.glob(os.path.join(self.datapath, "*"))

        self.videos = self.get_videos()
        self.n_videos = len(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        start = np.random.randint(0, video.shape[1] - (self.T + 1))
        end = start + self.T
        return video[:, start:end, ...].astype(np.float32)

    def get_videos(self):
        videos = [skvideo.io.vread(file) for file in self.files]
        # transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
        videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]

        return videos


class ToyPhysicsDataset(Dataset):
    def __init__(self, datapath, delta=1, train=True, resize=True, normalize=True):
        train_test = "train" if train else "test"
        self.T = config.video.frames
        self.resize = resize
        self.normalize = normalize

        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, f"{idx:06}.npy")

        vid = np.load(filename)
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.T + 1))
        end = start + self.T
        vid = vid[start:end]

        vid = (
            np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
                    for img in vid
                ]
            )
            if self.resize
            else vid
        )
        # vid = np.asarray([resize(img, (96, 96, nc)) for img in vid])
        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)
        # normalize -1 1
        vid = (vid - 0.5) / 0.5 if self.normalize else vid
        # vid = (vid - 0.5)/0.5

        return vid.astype(np.float32)


class ToyPhysicsDatasetNPZ(Dataset):
    def __init__(self, *, datapath, num_frames, delta=1, train=True):
        train_test = "train" if train else "test"
        self.num_frames = num_frames
        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, str(idx).zfill(5) + ".npz")

        vid = np.load(filename)["arr_0"]
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.num_frames + 1))
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != config.experiment.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        return vid.astype(np.float32), torch.tensor([])


class RealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_physics=10,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        jax.config.update("jax_enable_x64", True)

        if system_name is None:
            system_names = all_systems
            if system_color_constant:
                system_names = [x for x in system_names if "_COLORS" not in x]
            if system_friction:
                system_names = [x for x in system_names if "_FRICTION" in x]
        else:
            system_name = system_name.upper()
            if not system_color_constant:
                system_name += "_COLORS"
            if system_friction:
                system_name += "_FRICTION"

            assert system_name in all_systems, f"Unknown system {system_name}"
            system_names = [system_name]

        assert system_names, "No system selected"
        self.system_names = system_names

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_physics = ndim_physics
        self.img_size = img_size
        self.normalize = normalize

        self.generate_fn = {}  # Generate functions, keyed by system
        self.features = {}  # Fixed features across all trajectories, keyed by system

        for system_name in self.system_names:
            cls, config_ = getattr(datasets, system_name)
            config_dict = config_()

            # Tweak the physics parameters to our liking
            # TODO: Is this okay to do for speedup? Will this modify the characteristics of the experiment drastically?
            config_dict["image_resolution"] = img_size
            _physics_key = (
                system_name.replace("COLORS", "").replace("FRICTION", "").rstrip("_")
            )
            if system_physics_constant:
                config_dict |= constant_physics[_physics_key]
            else:
                config_dict |= variable_physics[_physics_key]

            obj = cls(**config_dict)

            f = functools.partial(
                datasets.generate_sample,
                system=obj,
                dt=0.05,  # Blanchette 2021
                # num_steps is always 1 less than the no. of samples we wish to generate
                num_steps=total_frames - 1,
                steps_per_dt=1,
            )

            self.generate_fn[system_name] = f
            self.features[system_name] = f(0)["other"]

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def _physics_vector_from_data(self, data):
        ndim_physics = self.ndim_physics
        if ndim_physics <= 0:
            return 0

        props = np.zeros((ndim_physics,))

        i = 0
        # note: dicts are ordered in py >= 3.7 so we have a deterministic order
        for k, v in data["other"].items():
            v = v.squeeze()
            if v.size == 1:
                props[i] = v.item()
                i += 1
                if i >= len(props):
                    return props
            else:
                for _v in v:
                    props[i] = _v
                    i += 1
                    if i >= len(props):
                        return props

        return props

    def __getitem__(self, item):
        system_name_index = np.random.choice(len(self.system_names))
        system_name = self.system_names[system_name_index]

        data = self.generate_fn[system_name](item)  # num_steps + 1, L, L, num_channels
        image = data["image"]
        # assert isinstance(image, jax.numpy.ndarray)
        # assert image.dtype == np.uint8

        vid = np.array(image / 255)
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - self.num_frames + 1)
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != self.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (
                            self.img_size,
                            self.img_size,
                            nc,
                        ),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        if self.normalize:
            vid = (vid - 0.5) / 0.5

        props = self._physics_vector_from_data(data)
        return vid.astype(np.float32), system_name_index, props


class HGNRealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_label=3,
        ndim_physics=10,
        ndim_color=0,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        self.system_names = all_systems_hgn
        self.n_systems = len(self.system_names)
        if system_name is None:
            self.system_index = None
        else:
            self.system_index = self.system_names.index(system_name)

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_label = ndim_label
        self.ndim_physics = ndim_physics
        self.ndim_color = ndim_color
        self.img_size = img_size
        self.normalize = normalize

        assert not bool(system_friction), "No friction supported yet"

        self.system_physics_constant = system_physics_constant
        self.system_color_constant = system_color_constant
        self.system_friction = system_friction

        self.system_name_mapping = {
            "mass_spring": "Spring",
            "pendulum": "Pendulum",
            "double_pendulum": "ChaoticPendulum",
            "two_body": "NObjectGravity",
            "three_body": "NObjectGravity",
        }

        self.system_embedding = torch.nn.Embedding(self.n_systems, self.ndim_label)

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def __getitem__(self, item):
        if self.system_index is None:
            system_index = np.random.choice(self.n_systems)
        else:
            system_index = self.system_index

        system_name = self.system_names[system_index]

        system_args_which = {True: constant_physics_hgn, False: variable_physics_hgn}[
            self.system_physics_constant
        ][system_name]

        system_args = {
            k: (v() if not isinstance(v, list) else [_v() for _v in v])
            for k, v in system_args_which.items()
        }

        system_name = self.system_name_mapping[system_name]
        system = EnvFactory.get_environment(system_name, **system_args)
        # We're not using self.total_frames here at all, since we only want self.num_frames from
        # the rollout, and the rollouts are randomly initialized anyway.

        vid = None
        colors = None
        # Rollouts are not guaranteed to give us self.num_frames in certain
        # cases where solve_ivp fails - keep trying till they do.
        while vid is None or vid.shape[0] != self.num_frames:
            vids, colors = system.sample_random_rollouts(
                number_of_frames=self.num_frames,
                delta_time=self.delta,
                number_of_rollouts=1,
                img_size=self.img_size,
                noise_level=0.1,
                radius_bound="auto",
                color=True,
                seed=None,
                constant_color=self.system_color_constant,
            )
            vid = vids[0]

        # transpose each video to (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        if self.normalize:
            vid = (vid - 0.5) / 0.5

        labels_and_props = torch.cat(
            (
                self.system_embedding(torch.tensor([system_index])).squeeze(),
                torch.tensor(system.physical_properties(vec_length=self.ndim_physics)),
            )
        )

        vid = vid.astype(np.float32)

        color_vec = torch.zeros(self.ndim_color)
        colors = torch.tensor(np.array(colors).flatten().astype(np.float32))[
            : self.ndim_color
        ]
        color_vec[: len(colors)] = colors

        return vid, labels_and_props, color_vec


class RealPendulumVideoDataset(Dataset):
    """
    Real-world double-pendulum video dataset (Mendeley z4hvxjgtbz).

    The source is a single continuous MP4 (`DP_free_drop_video.mp4`). On first
    use we preprocess it to a memory-mappable uint8 array of resized frames;
    subsequent runs reuse the cache.

    Returns (video, label_and_props, colors) matching HGNRealtimeDataset so the
    rest of the pipeline (Generator_I, Discriminator_V, etc.) is unchanged.
    When ndim_label + ndim_physics + ndim_color == 0 we return empty tensors
    for those fields (unconditional training).
    """

    def __init__(
        self,
        *,
        data_dir,
        video_filename="DP_free_drop_video.mp4",
        cache_filename=None,
        num_frames=30,
        img_size=96,
        ndim_channel=3,
        ndim_label=0,
        ndim_physics=0,
        ndim_color=0,
        stride=1,
        normalize=False,
        train=True,
        motion_percentile=0.0,
    ):
        self.data_dir = data_dir
        self.video_path = os.path.join(data_dir, video_filename)
        # Cache name includes img_size and channel count so switching grayscale
        # or resolution rebuilds cleanly instead of silently reusing a stale blob.
        if cache_filename is None:
            cache_filename = f"frames_{img_size}x{img_size}_{ndim_channel}ch.npy"
        self.cache_path = os.path.join(data_dir, cache_filename)

        self.num_frames = num_frames
        self.img_size = img_size
        self.ndim_channel = ndim_channel
        self.ndim_label = ndim_label
        self.ndim_physics = ndim_physics
        self.ndim_color = ndim_color
        self.stride = max(int(stride), 1)
        self.normalize = normalize
        self.train = train

        # Dummy Embedding so Experiment.save_epoch (which iterates over
        # model_names including "system_embedding") does not crash. It has no
        # effect on training since ndim_label is 0 here.
        self.system_embedding = torch.nn.Embedding(1, max(1, ndim_label))

        # DDP-safe cache build: only rank 0 writes; others wait on a barrier.
        self._build_cache_if_needed()
        self.frames = np.load(self.cache_path, mmap_mode="r")
        self.n_frames_total = self.frames.shape[0]

        clip_span = (self.num_frames - 1) * self.stride + 1
        if self.n_frames_total < clip_span:
            raise RuntimeError(
                f"Not enough frames in cache ({self.n_frames_total}) for a clip "
                f"of length {self.num_frames} with stride {self.stride}."
            )
        self.max_start = self.n_frames_total - clip_span

        # Build a list of clip start indices. When motion_percentile > 0 we
        # score every possible clip by the average frame-to-frame abs diff
        # across its frames and keep only those whose score is above the
        # chosen percentile of the distribution — that filters out the
        # pre-drop idle period and the long post-damping tail of the video.
        self.motion_percentile = float(motion_percentile)
        self.valid_starts = self._compute_valid_starts(self.motion_percentile)

    def _build_cache_if_needed(self):
        if os.path.exists(self.cache_path):
            return

        # Under DDP, only rank 0 builds; other ranks wait at a barrier so they
        # see the finished file when they resume.
        try:
            import torch.distributed as dist

            is_distributed = dist.is_available() and dist.is_initialized()
        except Exception:
            is_distributed = False

        if is_distributed and dist.get_rank() != 0:
            dist.barrier()
            return

        self._build_cache()

        if is_distributed:
            dist.barrier()

    def _build_cache(self):
        import cv2

        assert os.path.exists(self.video_path), f"Video not found: {self.video_path}"
        logger.info("Building frame cache %s from %s", self.cache_path, self.video_path)

        cap = cv2.VideoCapture(self.video_path)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-allocate the full array; uint8 keeps memory modest.
        if self.ndim_channel == 1:
            out = np.empty((n_total, self.img_size, self.img_size), dtype=np.uint8)
        else:
            out = np.empty(
                (n_total, self.img_size, self.img_size, self.ndim_channel),
                dtype=np.uint8,
            )

        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if self.ndim_channel == 1:
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                # cv2 returns BGR; convert to RGB to match the rest of the pipeline.
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            out[i] = resized
            i += 1
            if i % 5000 == 0:
                logger.info("  processed %d / %d frames", i, n_total)
        cap.release()

        out = out[:i]  # in case the reported frame count was off
        # np.save appends `.npy` if the path lacks it, so name the tmp file
        # with the same extension to guarantee atomic replace.
        tmp_path = self.cache_path + ".tmp.npy"
        np.save(tmp_path, out)
        os.replace(tmp_path, self.cache_path)
        logger.info("Cached %d frames to %s", out.shape[0], self.cache_path)

    def _compute_valid_starts(self, percentile):
        """Return the array of clip start indices eligible for sampling.

        If `percentile <= 0` every start is eligible (no filtering). Otherwise
        each possible start is scored by the mean |frame[t+stride] - frame[t]|
        over the clip's frames, and only starts whose score is at or above the
        chosen percentile of the global score distribution are kept. This
        focuses training on the actively-swinging portion of the video and
        away from the still/damped segments.
        """
        n_starts = self.max_start + 1
        if percentile <= 0 or n_starts <= 1:
            return np.arange(n_starts, dtype=np.int64)

        frames = self.frames
        # Per-interval motion profile at the sampling stride. Subsample
        # densely (every `step` frames) to keep this fast on long videos.
        step = max(1, self.stride)
        pair_indices = np.arange(0, self.n_frames_total - step, step)
        diffs = np.empty(len(pair_indices), dtype=np.float32)
        for i, t in enumerate(pair_indices):
            a = np.asarray(frames[t]).astype(np.float32)
            b = np.asarray(frames[t + step]).astype(np.float32)
            diffs[i] = np.abs(a - b).mean()

        # For each possible clip start, compute the mean of the `num_frames-1`
        # consecutive diffs that fall inside the clip span.
        scores = np.empty(n_starts, dtype=np.float32)
        cumsum = np.concatenate(([0.0], np.cumsum(diffs).astype(np.float64)))
        win = self.num_frames - 1  # diffs per clip
        for s in range(n_starts):
            i0 = s // step
            i1 = min(i0 + win, len(diffs))
            if i1 <= i0:
                scores[s] = 0.0
            else:
                scores[s] = (cumsum[i1] - cumsum[i0]) / (i1 - i0)

        thresh = np.percentile(scores, percentile)
        mask = scores >= thresh
        valid = np.where(mask)[0].astype(np.int64)
        kept = len(valid)
        logger.info(
            "RealPendulumVideoDataset: motion_percentile=%.1f → kept %d / %d "
            "starts (threshold=%.4f, score range=%.4f-%.4f)",
            percentile,
            kept,
            n_starts,
            float(thresh),
            float(scores.min()),
            float(scores.max()),
        )
        if kept == 0:
            raise RuntimeError(
                f"motion_percentile={percentile} filtered out every clip"
            )
        return valid

    def __len__(self):
        # Follow the convention of HGNRealtimeDataset (50k / 10k) so the
        # epoch-based schedule in the .ini works identically. Every
        # __getitem__ picks a new random clip anyway.
        return 50_000 if self.train else 10_000

    def __getitem__(self, idx):
        start = int(self.valid_starts[np.random.randint(0, len(self.valid_starts))])
        end = start + (self.num_frames - 1) * self.stride + 1
        # Cache is (T, H, W) for grayscale, (T, H, W, C) for multi-channel.
        clip = np.asarray(self.frames[start:end:self.stride])
        if clip.ndim == 3:  # grayscale → add channel dim
            clip = clip[..., None]

        vid = clip.astype(np.float32) / 255.0
        if self.normalize:
            vid = (vid - 0.5) / 0.5

        # (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        label_and_props = torch.zeros(
            self.ndim_label + self.ndim_physics, dtype=torch.float32
        )
        colors = torch.zeros(self.ndim_color, dtype=torch.float32)

        return vid, label_and_props, colors
