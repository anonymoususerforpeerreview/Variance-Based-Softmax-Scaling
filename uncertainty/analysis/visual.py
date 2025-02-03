import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST
import pickle

from shared.utils import get_analysis_id

try:
    import tikzplotlib
except ImportError:
    print("tikzplotlib not installed. Install it using `pip install tikzplotlib`.")


class Visual:
    class DiscrDistr:
        @staticmethod
        def correct_vs_wrong(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].cpu().numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].cpu().numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                ax.hist(variances_filtered, bins=25, color=colors[i], alpha=0.5, label=labels[i])

            ax.set_xlabel("Variance")
            ax.set_ylabel("Count")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            plt.show()
            plt.close(fig)

            return fig

        @staticmethod
        def correct_vs_wrong_normalized(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].cpu().numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].cpu().numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                ax.hist(variances_filtered, bins=25, color=colors[i], alpha=0.5, label=labels[i], density=True)

            ax.set_xlabel("Variance")
            ax.set_ylabel("Density")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            plt.show()
            plt.close(fig)

            return fig

        @staticmethod
        def distributions_under_diff_noise_levels(h: dict, noise_levels, noise_type, distributions, labels) -> list[
            plt.Figure]:
            """In: distributions numpy array shape: (num_noise_levels, batch_size, num_classes) """
            resulting_figs = []

            for noise_idx in range(distributions.shape[0]):
                for sample_idx in range(distributions.shape[1]):
                    distr = distributions[noise_idx, sample_idx]  # (num_classes)

                    # Permute the distribution such that first class is the correct class
                    correct_class_idx = labels[noise_idx, sample_idx]
                    distr = np.roll(distr, -correct_class_idx)  # (num_classes)

                    # y_min = min(y_min, distr.min())
                    # y_max = max(y_max, distr.max())

                    num_classes = len(distr)

                    # just show one distribution in a figure
                    fig, ax = plt.subplots()
                    ax.bar(range(num_classes), distr, color=['red'] + ['blue'] * (num_classes - 1))
                    # ax.set_ylim(y_min, y_max)
                    # ax.set_ylim(0, 1)
                    ax.set_title(f"Noise Level {noise_levels[noise_idx]}, Sample {sample_idx}")

                    save_path = f"{h['log_path']}/distribs_under_noise_sample={sample_idx}/"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if save_path is not None:
                        fig.savefig(f"{save_path}/distr_noise_{noise_idx}.pdf")
                        try:
                            tikzplotlib.save(f"{save_path}/distr_noise_{noise_idx}.tex")
                        except Exception as e:
                            print(f"Error when saving to tikz: {e}. Storing fig using pickle instead.")
                            with open(f"{save_path}/sample_{sample_idx}.pkl", 'wb') as f:
                                pickle.dump(fig, f)

                    # append
                    resulting_figs.append(fig)
                    plt.show()
                    plt.close(fig)

            return resulting_figs

    class ContinDistr:
        @staticmethod
        def correct_vs_wrong(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].cpu().numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].cpu().numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                sns.kdeplot(variances_filtered, color=colors[i], label=labels[i])

            ax.set_xlabel("Variance")
            ax.set_ylabel("Density")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            plt.show()
            plt.close(fig)

            return fig

    class LogitDistr:
        @staticmethod
        def logits_under_diff_noise_levels(h: dict, noise_levels, noise_type, logits, labels) -> list[plt.Figure]:
            if h['log_path'] is not None:
                analysis_id: str = get_analysis_id(h)
                # remove special characters {, }, :, ", '. characters such as space, /, \, etc. can remain
                analysis_id = analysis_id.translate(str.maketrans("", "", "{}:\"'"))
                save_path = f"{h['log_path']}/{analysis_id}/logits_under_diff_noise_levels/{noise_type}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            else:
                save_path = None

            print(f"Displaying logits distribution under different noise levels for {noise_type} noise.")
            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            """
            In:
            - logits numpy array shape: (num_noise_levels, batch_size, num_views, H'*W', num_classes)
            - labels numpy array shape: (num_noise_levels, batch_size)

            Plot the distribution of logits under different noise levels as histograms.
            Per noise level, creates a graph containing 8*8 subplots, each subplot showing the distribution of logits.
            Rows represent distributions of different samples (part of batch) and columns represent different temporal resolutions (H'*W').
            """
            nb_rows = 5  # max: len(noise_levels)
            nb_cols = min(8, logits.shape[3])  # nb of temporal resolutions

            # pick `nb_cols` indices between 0 and logits.shape[3] to show the distribution of logits at these. Take values that are not too close to each other
            temporal_resolution_indices = np.linspace(start=0, stop=logits.shape[3] - 1, num=nb_cols, dtype=int)
            assert len(temporal_resolution_indices) == nb_cols

            resulting_figs = []

            for batch_idx in range(logits.shape[1]):  # iterate over samples (batch)
                fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(40, 40))
                fig.suptitle(f"Logits for Sample {batch_idx} (y: Noise Level, x: Temporal Resolution)")

                # 5 first noise levels as those are the most interesting
                for noise_level_idx, noise_level in enumerate(noise_levels[:nb_rows]):
                    y_min, y_max = float('inf'), float('-inf')
                    for col, temporal_idx in enumerate(
                            temporal_resolution_indices):  # iterate over temporal resolutions.
                        logit_distr = logits[noise_level_idx, batch_idx, 0, temporal_idx]  # (num_classes)

                        # Permute the logit distribution such that first class is the correct class
                        correct_class_idx = labels[noise_level_idx, batch_idx]
                        logit_distr = np.roll(logit_distr, -correct_class_idx)  # (num_classes)

                        # Only show the first 20 classes
                        logit_distr = logit_distr[:20]  # in case of too many classes, only show the first 20
                        y_min = min(y_min, logit_distr.min())
                        y_max = max(y_max, logit_distr.max())

                        ax = axes[noise_level_idx, col]
                        num_classes = len(logit_distr)

                        # Color the first bar as this is now the correct class
                        ax.bar(range(num_classes), logit_distr, color=['red'] + ['blue'] * (num_classes - 1))

                        ax.set_ylim(y_min, y_max)
                        ax.set_title(f"Noise Level {noise_level}, Temporal {temporal_idx}")

                plt.tight_layout()
                if save_path is not None:
                    fig.savefig(f"{save_path}/sample_{batch_idx}.pdf")
                    try:
                        tikzplotlib.save(f"{save_path}/sample_{batch_idx}.tex")
                    except Exception as e:
                        print(f"Error when saving to tikz: {e}. Storing fig using pickle instead.")
                        with open(f"{save_path}/sample_{batch_idx}.pkl", 'wb') as f:
                            pickle.dump(fig, f)

                plt.show()
                resulting_figs.append(fig)
                plt.close(fig)

            # store logits, labels, noise_levels, temporal_resolution_indices, noise_type to disk for optional later use (as a pickle file)
            if save_path is not None:
                np.save(f"{save_path}/logits.npy", logits)
                np.save(f"{save_path}/labels.npy", labels)
                np.save(f"{save_path}/noise_levels.npy", noise_levels)
                np.save(f"{save_path}/temporal_resolution_indices.npy", temporal_resolution_indices)
                np.save(f"{save_path}/noise_type.npy", noise_type)

            return resulting_figs

    @staticmethod
    def plot_line(x_values, y_values, title, x_label, y_label, y_min: float = None, y_max: float = None):
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        plt.show()
        fig = plt.gcf()
        plt.close(fig)  # Avoids third plot from showing up when calling plot_line multiple times
        return fig

    @staticmethod
    def plot_x_y_scatter(x_values, y_values, title, x_label, y_label):
        plt.scatter(x_values, y_values)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        fig = plt.gcf()
        plt.close(fig)

        return fig

    @staticmethod
    def show_transformations(h: dict, dataset_path: str, transformations: list[callable], num_images: int = 1):
        # dispay the different subsampling methods, same image repeated 3 times
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(dataset_path, train=True, download=True, transform=transform)

        for im_idx in range(num_images):
            x, y = mnist_train[im_idx]
            for method in transformations:
                # create 3 images and display them side by side
                imgs = [method(x, **h['redundancy_method_params']) for _ in range(3)]  # T.retrieve_transforms(h)
                imgs = torch.cat(imgs, dim=2)
                Visual.plot_img(imgs, title=f"Image {im_idx} - {method.__name__}")

        return plt.gcf()

    @staticmethod
    def plot_img(img, title: str) -> plt.Figure:
        if isinstance(img, plt.Figure):
            img.show()
            return img
        if isinstance(img, Tensor):
            # if shape is (3, H, W), convert to (H, W, 3)
            if img.dim() == 3 and img.size(0) == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()  # Move tensor to CPU before converting to NumPy
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

        return plt.gcf()

    @staticmethod
    def plot_signal(x: Union[Tensor, np.array], title: str) -> plt.Figure:
        """
        Plot a signal with multiple channels.
        """
        # x: (C, L)
        num_channels = x.size(0)
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))

        if num_channels == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(x[i].numpy())
            ax.set_title(f"{title} - Channel {i + 1}")

        plt.tight_layout()
        plt.show()

        return plt.gcf()


if __name__ == "__main__":
    # 20, 8, 1, 75, 20
    # logit_distrs = np.random.rand(20, 8, 1, 75, 20) - 0.2  # (num_noise_levels, batch_size, num_views, H'*W', num_classes)
    nb_classes = 251
    nb_samples = 5
    nb_noise_levels = 20
    labels = np.random.randint(0, nb_classes, (nb_noise_levels, nb_samples))  # (num_noise_levels, batch_size)

    h = {'log_path': "./temp/",
         'model_name': '',
         'UQ_predictor': '',
         'predictor_params': '',
         'final_activation': '',
         'redundancy_method': '',
         'num_views': '',
         'redundancy_method_params': ''}
    # Visual.LogitDistr.logits_under_diff_noise_levels(h, np.arange(20), "Gaussian", logit_distrs, labels)

    logit_distrs = np.random.rand(nb_noise_levels, nb_samples,
                                  nb_classes)  # (num_noise_levels, batch_size, num_classes)

    # softmax
    logit_distrs = np.exp(logit_distrs) / np.exp(logit_distrs).sum(axis=-1, keepdims=True)

    Visual.DiscrDistr.distributions_under_diff_noise_levels(h, np.linspace(0, 1, nb_noise_levels),
                                                            "Gaussian", logit_distrs, labels)


    # Example of how to save a pickled figure as a tikz file
    def save_fig_as_tikz(pkl_file_path, tikz_file_path):
        # in case tikzplotlib is not installed, retrieve in the future by loading the pickled file:
        with open(pkl_file_path, 'rb') as f:
            fig = pickle.load(f)

        tikzplotlib.save(tikz_file_path, figure=fig)

    # save_fig_as_tikz('./logits_under_diff_noise_levels/Gaussian/sample_0.pkl',
    # './logits_under_diff_noise_levels/Gaussian/sample_0.tex')
