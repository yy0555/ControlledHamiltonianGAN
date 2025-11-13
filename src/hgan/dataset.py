import os
import glob
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
import matplotlib.pyplot as plt


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
    
    def compute_rmse(self, traj_gen, traj_real, eps=1e-8):
        traj_gen = np.array(traj_gen)
        traj_real = np.array(traj_real)

        diff = traj_gen - traj_real
        rmse = np.sqrt(np.mean(diff ** 2))

        return rmse
    
    def plot_2d_trajectory_comparison(self, traj, mask, real, save_path, label1='Fake', label2='Real', system_name=None):
        import os
        valid_idx = np.where(mask > 0)[0]
        N_eff = len(valid_idx)
        T = traj.shape[0]
        T_real = real.shape[0]

        # 优先用传入的 system_name，否则用 self.system_name
        sys_name = system_name if system_name is not None else getattr(self, 'system_name', 'Unknown')

        plt.figure(figsize=(6, 6))
        colors = plt.cm.tab10.colors
        rmse_list = []
        if sys_name == "Spring":
            # 1D系统，所有粒子画在一张图上，x随时间变化
            time = np.arange(T)
            for i, idx in enumerate(valid_idx):
                plt.plot(time, traj[:, 0, idx], label=f'{label1} Particle {idx+1}', color=colors[i % len(colors)], linestyle='-')
                plt.plot(time, real[:, 0, idx], label=f'{label2} Particle {idx+1}', color=colors[i % len(colors)], linestyle='--')
                fake_traj = traj[:, 0, idx]
                real_traj = real[:, 0, idx]
                if T == T_real:
                    rmse = self.compute_rmse(fake_traj, real_traj, eps=1e-8)
                    rmse_list.append(rmse)
            plt.xlabel('Time')
            plt.ylabel('x')
        else:
            # 2D系统，所有粒子画在一张图上，xy轨迹
            for i, idx in enumerate(valid_idx):
                plt.plot(traj[:, 0, idx], traj[:, 1, idx], label=f'{label1} Particle {idx+1}', color=colors[i % len(colors)], linestyle='-')
                plt.plot(real[:, 0, idx], real[:, 1, idx], label=f'{label2} Particle {idx+1}', color=colors[i % len(colors)], linestyle='--')
                fake_traj = traj[:, :, idx]
                real_traj = real[:, :, idx]
                if T == T_real:
                    rmse = self.compute_rmse(fake_traj, real_traj, eps=1e-8)
                    rmse_list.append(rmse)
            plt.xlabel('x')
            plt.ylabel('y')

        if T == T_real and rmse_list:
            mean_rmse = np.mean(rmse_list)
            title = f"{sys_name} (RMSE={mean_rmse:.4f})"
        else:
            title = f"{sys_name} (T mismatch)"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Save trajectory data
        if system_name is None:
            save_data_path = os.path.splitext(save_path)[0] + '_data.npz'
            np.savez(save_data_path, fake=traj, real=real, mask=mask, system_name=sys_name)

    def _compute_momentum(self, q_t, q_t1):
        mass = np.atleast_1d(self.system_args["mass"])
        p = mass[:, None] * (q_t1 - q_t) / self.delta
        return p.reshape(q_t.shape)

    def pad_traj(self, traj, n_max=3):
        n_particles = traj.shape[2]
        T = traj.shape[0]
        padded = torch.zeros((T, 2, n_max), dtype=torch.float32)
        padded[:, :, :n_particles] = torch.tensor(traj, dtype=torch.float32)

        mask = torch.zeros(n_max, dtype=torch.float32)
        mask[:n_particles] = 1.0

        return padded, mask

    def __getitem__(self, item):
        if self.system_index is None:
            system_index = np.random.choice(self.n_systems)
        else:
            system_index = self.system_index

        system_name = self.system_names[system_index]

        system_args_which = {True: constant_physics_hgn, False: variable_physics_hgn}[
            self.system_physics_constant
        ][system_name]

        self.system_args = {
            k: (v() if not isinstance(v, list) else [_v() for _v in v])
            for k, v in system_args_which.items()
        }

        self.system_name = self.system_name_mapping[system_name]
        system = EnvFactory.get_environment(self.system_name, **self.system_args)
        # We're not using self.total_frames here at all, since we only want self.num_frames from
        # the rollout, and the rollouts are randomly initialized anyway.

        colors = None
        rollout = None
        # Rollouts are not guaranteed to give us self.num_frames in certain
        # cases where solve_ivp fails - keep trying till they do.
        while rollout is None or rollout.shape[0] != self.num_frames:
            rollout = system.sample_random_rollouts(
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
        padded, mask = self.pad_traj(rollout)

        labels_and_props = torch.cat(
            (
                self.system_embedding(torch.tensor([system_index])).squeeze(),
                torch.tensor(system.physical_properties(vec_length=self.ndim_physics)),
            )
        )

        return padded, mask, labels_and_props
    
    def get_system_ids(self, label_and_props):
        """
        label_and_props: torch.Tensor (B, n_label+params)
        返回 system_id: (B,) 整数ID
        """
        system_code = label_and_props[:, :3]
        unique_codes = {tuple(code): idx for idx, code in enumerate(np.unique(system_code, axis=0))}
        system_id = np.array([unique_codes[tuple(code)] for code in system_code])
        return system_id
    


    def comparison(self, trajectory, mask, folder, epoch, prefix, frame_idx=0):
        os.makedirs(folder, exist_ok=True)
        filename = f"{prefix}{epoch:0>6}"
        file_path = os.path.join(folder, f"{filename}.jpg")

        system = EnvFactory.get_environment(self.system_name, **self.system_args)
        q = system.extract_q(trajectory, mask, frame_idx=frame_idx)
        q1 = system.extract_q(trajectory, mask, frame_idx=frame_idx+1)
        p = self._compute_momentum(q,q1)
        
        rollout = system.calculate_fixed_rollout(q, p, number_of_frames=self.num_frames, delta_time=self.delta)

        self.plot_2d_trajectory_comparison(trajectory, mask, rollout, file_path)
   
