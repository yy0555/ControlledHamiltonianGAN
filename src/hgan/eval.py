import os
import sys
import argparse
import logging
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from sklearn.manifold import TSNE
from hgan.configuration import load_config
from hgan.experiment import Experiment


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder where results will be generated",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Process every nth epoch checkpoint encountered (default 1)",
    )
    parser.add_argument(
        "--generated-videos-timeslots",
        type=int,
        default=8,
        help="Number of timeslots to include in generated videos pngs (default 8)",
    )
    parser.add_argument(
        "--generated-videos-samples",
        type=int,
        default=3,
        help="Number of samples to include in generated videos pngs (default 3)",
    )
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=1024,
        help="Number of latent samples to generate for TSNE embedding (default 1024)",
    )
    parser.add_argument(
        "--calculate-fvd",
        dest="calculate_fvd",
        action="store_true",
        default=False,
        help="Calculate fvd score for every processed epoch (expensive operation!)",
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=16,
        help="Number of real/fake videos to consider for fvd calculation (default 16)",
    )
    parser.add_argument(
        "--fvd-on-cpu",
        action="store_true",
        default=False,
        help="Whether to run FVD on cpu (for low memory GPUs; default False)",
    )
    return parser


def qualitative_results_img(
    experiment, png_path, fake=True, timeslots=8, samples=3, title=""
):
    data = experiment.get_fake_data() if fake else experiment.get_real_data()

    # (batch_size, nc, T, img_size, img_size) => (batch_size, T, img_size, img_size, nc)
    videos = data["videos"].permute(0, 2, 3, 4, 1)
    # Normalize from [-1, 1] to [0, 1]
    videos = videos / 2 + 0.5
    videos = videos.detach().cpu().numpy().squeeze()

    videos = videos[:samples, :timeslots, :, :, :]
    # Create a ndarray of video frames: timeslots, then samples_per_timeslot
    videos = videos.reshape((-1, *videos.shape[2:]))

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title)
    # A grid in which each column represents a timeslot and each row a different instantiation of the
    # video at that time slot
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(samples, timeslots),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, videos):
        ax.axis("off")
        ax.imshow(im)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig=fig)


def qualitative_results_latent(
    experiment, batch_size, png_path, perplexity_values=(2, 5, 30, 50, 100), title=""
):
    Z, _, _, _ = experiment.get_latent_sample(
        batch_size=batch_size,
        n_frames=1,
    )  # shape (batch_size, n_frames, |ndim_q + ndim_p + ndim_content + ndim_label|, 1, 1)

    X = [
        Z[i, 0, : experiment.ndim_q].data.cpu().numpy().squeeze()
        for i in range(batch_size)
    ]
    X = np.asarray(X).reshape(-1, experiment.ndim_q)  # shape (batch_size, ndim_q)

    fig, axs = plt.subplots(
        ncols=len(perplexity_values), nrows=1, figsize=(20, 6), layout="constrained"
    )
    fig.suptitle(title)

    for i, p in enumerate(perplexity_values):
        tsne = TSNE(n_components=2, perplexity=p, init="random").fit_transform(X)
        axs[i].plot(tsne[:, 0], tsne[:, 1], ".")
        axs[i].set_title(f"Perplexity {p}")
        axs[i].axis("off")

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig=fig)


def main(*args):
    args = get_parser().parse_args(args)
    config = load_config(args.config_path)

    output_folder = args.output_folder
    config.save(output_folder)

    experiment = Experiment(config)
    experiment.eval()

    rnn = experiment.rnn
    hnn = rnn.hnn
    rnn_input_shape = (
        experiment.ndim_epsilon + experiment.ndim_label + experiment.ndim_physics,
    )
    energy_file = os.path.join(output_folder, "energy.txt")

    saved_epochs = experiment.saved_epochs()
    for epoch in saved_epochs[:: args.every_nth]:
        logger.info(f"Processing epoch {epoch}")
        experiment.load_epoch(epoch)

        logger.info("  Calculating Energy")
        noise = torch.randn(*rnn_input_shape).to(experiment.device)
        z = rnn.phase_space_map(noise)
        labels_and_physical_props = noise[experiment.ndim_epsilon :]
        hnn_input = torch.cat((z, labels_and_physical_props))
        hnn_output = hnn(hnn_input)
        energy = float(hnn_output)

        with open(energy_file, "a") as f:
            f.write(f"epoch={epoch}, energy={energy}\n")

        logger.info("  Generating Videos Image")
        qualitative_results_img(
            experiment,
            f"{output_folder}/videos_{epoch:06d}.png",
            timeslots=args.generated_videos_timeslots,
            samples=args.generated_videos_samples,
            title=f"Epoch {epoch}",
        )

        logger.info("  Generating Latent Features Image")
        qualitative_results_latent(
            experiment=experiment,
            batch_size=args.latent_batch_size,
            png_path=f"{output_folder}/config_{epoch:06d}.png",
            title=f"Epoch {epoch}",
        )

        if args.calculate_fvd:
            logger.info("  Calculating FVD Score")
            fvd_device = "cpu" if args.fvd_on_cpu else experiment.device
            fvd = experiment.fvd(device=fvd_device, max_videos=args.fvd_batch_size)
            fvd_score_file = os.path.join(output_folder, "fvd_scores.txt")
            with open(fvd_score_file, "a") as f:
                f.write(f"epoch={epoch}, fvd={fvd}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
