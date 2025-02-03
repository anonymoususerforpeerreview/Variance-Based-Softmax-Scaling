import torch.cuda
from torch.utils.data import DataLoader

from shared.abstract_model import ANN
from shared.data.data_loader import data_loaders
from shared.data.dataset_meta import DatasetMeta as M
from shared.data.noise import NoiseFactory
from shared.decorators import init_decorator, timer_decorator, wandb_resume_but_new_run_decorator
from shared.hyperparameters import Hyperparameters
from shared.wandb import W  # wandb wrapper
from uncertainty.analysis.metrics import Metrics
from uncertainty.analysis.uq import UQ
from uncertainty.analysis.visual import Visual
from uncertainty.uq_through_redundancy.classifier_factory import ClassifierFactory
from uncertainty.uq_through_redundancy.uq_predictor import UQPredictor, UQPredictorFactory


def compute_metrics(h: dict, test_loader: DataLoader, uq_predictor: UQPredictor):
    accuracy = Metrics.compute_accuracy(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])
    W.log_scalar(h, "Accuracy", accuracy)

    nll = Metrics.compute_nll(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])
    W.log_scalar(h, "NLL", nll)

    brier_score = Metrics.compute_brier_score(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])
    W.log_scalar(h, "Brier Score", brier_score)

    cross_entropy = Metrics.compute_cross_entropy(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])
    W.log_scalar(h, "Cross Entropy", cross_entropy)

    ece = Metrics.compute_ece(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])
    W.log_scalar(h, "ECE", ece)  # Expected Calibration Error


def variance_error(key: str, test_loader: DataLoader, h: dict, uq_predictor: UQPredictor):
    if h['num_views'] > 1:
        raise NotImplementedError("Variance-Error distribution is only implemented for num_views=1")
        print("Computing variance-error distribution...")
        # (batch_size, 2) where 1st column is variance and 2nd is 1 or 0 if the prediction is correct or not
        res = UQ.variances_vs_accuracy_per_input_img(uq_predictor, test_loader, limit_batches=h['limit_test_batches'])

        print(f"Variance-Error Distribution Shape: {res.shape}")

        W.log_im(
            h,
            Visual.DiscrDistr.correct_vs_wrong(res.detach()),
            f"{key}/DiscrDistr")
        W.log_im(
            h,
            Visual.ContinDistr.correct_vs_wrong(res.detach()),
            f"{key}/ContinDistr")

        W.log_im(
            h,
            Visual.DiscrDistr.correct_vs_wrong_normalized(res.detach()),
            f"{key}/DiscrDistr_normalized")


def _display_samples(h, key, noise_type, noisy_img_samples):
    if M.display_samples(h):  # e.g. for radio will return False
        W.log_im(
            h,
            noisy_img_samples,
            f"{key}__{noise_type}/Noisy_images_used_for_entropy_calculation")


def _display_kernel_accuracies_and_logits(h, key, noise_type, uq_predictor, test_loader):
    # ############################################################################################################
    # Accuracies for Kernel Based Inputs
    if h['UQ_predictor'] in ['StdBasedSoftmaxRelaxation', 'StdBasedSoftmaxRelaxationLearnedParams']:
        kernel_size = h['predictor_params']['kernel_size']

        ### KERNEL BASED INPUTS
        noise_levels, mean_accuracies_kernel_based_inputs = \
            UQ.apply_f_to_all_noise_levels(
                function=lambda noise_level: UQ.compute_accuracy_kernel_based_inputs(
                    noise_level, noise_type,
                    kernel_size, uq_predictor,
                    test_loader, limit_batches=h['limit_test_batches']))

        W.log_im(
            h,
            Visual.plot_line(
                noise_levels, mean_accuracies_kernel_based_inputs,
                f"Accuracy of Kernel Based Inputs under Distributional Shift ({noise_type} noise)",
                "Noise Level", "Mean Accuracy"),
            f"{key}__{noise_type}_fig/Accuracy_kernel_based_inputs_vs_noise_level_fig")

        W.log_x_y(h, noise_levels, mean_accuracies_kernel_based_inputs,
                  f"{key}__{noise_type}/Accuracy_kernel_based_inputs_vs_noise_level",
                  x_label="Noise Level (std dev of noise)",
                  y_label="Mean Accuracy of Kernel Based Inputs")

        ### Display some of the logits under different noise levels
        noise_levels, logits, labels = UQ.apply_f_to_all_noise_levels(
            function=lambda noise_level: UQ.compute_logits_kernel_based_inputs(
                noise_level, noise_type, kernel_size, uq_predictor, test_loader,
                limit_batches=h['limit_test_batches'],
                n_elements=5  # number of elements to display
            ))
        # logits shape: (num_noise_levels, batch_size, num_views, H'*W', num_classes)
        # labels shape: (num_noise_levels, batch_size)
        # Visual.LogitDistr.logits_under_diff_noise_levels(h, noise_levels, noise_type, logits, labels)
        W.log_ims(h,
                  Visual.LogitDistr.logits_under_diff_noise_levels(h, noise_levels, noise_type, logits, labels),
                  f"{key}__{noise_type}/Logits_under_diff_noise_levels")


def entropy_ood(key: str, test_loader: DataLoader, h: dict, uq_predictor: UQPredictor):
    for noise_type in NoiseFactory.SUPPORTED_NOISE_TYPES():  # SUPPORTED_NOISE_TYPES[:1]:  # For only Gaussian noise
        print(f"Computing entropy ({noise_type})...")
        noise_levels, mean_entropies, mean_accuracies, mean_stds, noisy_img_samples = \
            UQ.apply_f_to_uq_score_all_noise_levels(uq_predictor, test_loader,
                                                    function=UQ.compute_entropy,
                                                    noise_type=noise_type,
                                                    limit_batches=h['limit_test_batches'],
                                                    is_1d_signal=M.is_1d_signal(h),
                                                    display_samples=M.display_samples(h))
        print(f"Entropy len: {len(noise_levels)}, Noise levels: {noise_levels}, "
              f"Mean entropies: {mean_entropies},  Mean accuracies: {mean_accuracies}, Mean stds: {mean_stds}")

        # ENTROPY
        W.log_im(
            h,
            Visual.plot_line(
                noise_levels, mean_entropies,
                f"Entropy under Distributional Shift ({noise_type} noise)",
                "Noise Level", "Mean Entropy"),
            f"{key}__{noise_type}_fig/Entropy_vs_noise_level_fig")

        W.log_x_y(h, noise_levels, mean_entropies,
                  f"{key}__{noise_type}/Entropy_vs_noise_level",
                  x_label="Noise Level (std dev of noise)",
                  y_label="Mean Entropy (Model Uncertainty)")

        # ENTROPY NORMALIZED
        sum = mean_entropies.sum()
        mean_entropies_normalized = mean_entropies / sum
        W.log_im(
            h,
            Visual.plot_line(
                noise_levels, mean_entropies_normalized,
                f"Entropy under Distributional Shift ({noise_type} noise) Normalized",
                "Noise Level", "Mean Entropy Normalized",
                y_min=0, y_max=0.12
            ),
            f"{key}__{noise_type}_fig/Entropy_vs_noise_level_normalized_fig")

        W.log_x_y(h, noise_levels, mean_entropies_normalized,
                  f"{key}__{noise_type}/Entropy_vs_noise_level_normalized",
                  x_label="Noise Level (std dev of noise)",
                  y_label="Mean Entropy Normalized (Model Uncertainty)")

        # ACCURACY
        W.log_im(
            h,
            Visual.plot_line(
                noise_levels, mean_accuracies,
                f"Accuracy under Distributional Shift ({noise_type} noise)",
                "Noise Level", "Mean Accuracy"),
            f"{key}__{noise_type}_fig/Accuracy_vs_noise_level_fig")
        #
        W.log_x_y(h, noise_levels, mean_accuracies, f"{key}__{noise_type}/Accuracy_vs_noise_level",
                  x_label="Noise Level (std dev of noise)",
                  y_label="Mean Accuracy")

        # STD
        W.log_im(
            h,
            Visual.plot_line(
                noise_levels, mean_stds,
                f"STD under Distributional Shift ({noise_type} noise)",
                "Noise Level", "Mean STD"),
            f"{key}__{noise_type}_fig/STD_vs_noise_level_fig")

        W.log_x_y(h, noise_levels, mean_stds, f"{key}__{noise_type}/STD_vs_noise_level",
                  x_label="Noise Level (std dev of noise)",
                  y_label="Mean STD")

        _display_samples(h, key, noise_type, noisy_img_samples)

        _display_kernel_accuracies_and_logits(h, key, noise_type, uq_predictor, test_loader)


def confidence_vs_avg_accuracy_line(key: str, uq_predictor: UQPredictor, test_loader: DataLoader, h: dict):
    # uncertainy_per_avg_accuracy_line
    uncertainty_bins, avg_accuracies, num_datapoints_per_bin = \
        UQ.uncertainty_per_avg_accuracy_line(uq_predictor, test_loader,
                                             num_bins=10,
                                             limit_batches=h['limit_test_batches'])

    W.log_x_y(h, uncertainty_bins, avg_accuracies, f"{key}/Uncertainty_vs_Avg_Accuracy",
              x_label="Uncertainty",
              y_label="Average Accuracy")

    W.log_x_y(h, uncertainty_bins, num_datapoints_per_bin, f"{key}/Uncertainty_vs_Num_Datapoints ",
              x_label="Uncertainty",
              y_label="Number of Datapoints")


def display_predicted_distributions_ood(key: str, test_loader: DataLoader, h: dict, uq_predictor: UQPredictor):
    noise_type = NoiseFactory.SUPPORTED_NOISE_TYPES()[0]  # For only Gaussian noise
    print(f"Computing predicted distributions ({noise_type})...")

    start, stop, num = 0, 0.7, 60

    ### Display some of the distributions under different noise levels
    noise_levels, distributions, labels = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: UQ.compute_predicted_distributions(
            noise_level, noise_type, uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            n_elements=5  # number of elements to display per noise level
        ), start=start, stop=stop, num=num)

    W.log_ims(h,
              # Out: (num_noise_levels, batch_size, num_classes)
              Visual.DiscrDistr.distributions_under_diff_noise_levels(
                  h, noise_levels, noise_type, distributions, labels),
              f"{key}__{noise_type}/distributions_under_diff_noise_levels")


def metrics_under_distributional_shift(key: str, test_loader: DataLoader, h: dict, uq_predictor: UQPredictor):
    # ece, accuracy, brier_score, entropy
    noise_type = NoiseFactory.SUPPORTED_NOISE_TYPES()[0]  # For only Gaussian noise
    print(f"Computing metrics under distributional shift ({noise_type})...")

    start, stop, num = 0, 0.7, 60
    noise_levels, ece = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: Metrics.compute_ece(
            uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            noise_type=noise_type,
            noise_level=noise_level),
        start=start, stop=stop, num=num)

    W.log_x_y(h, noise_levels, ece, f"{key}/ECE_vs_noise_level",
              x_label="Noise Level (std dev of noise)",
              y_label="ECE")

    noise_levels, accuracy = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: Metrics.compute_accuracy(
            uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            noise_type=noise_type,
            noise_level=noise_level),
        start=start, stop=stop, num=num)

    W.log_x_y(h, noise_levels, accuracy, f"{key}/Accuracy_vs_noise_level",
              x_label="Noise Level (std dev of noise)",
              y_label="Accuracy")

    noise_levels, brier_score = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: Metrics.compute_brier_score(
            uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            noise_type=noise_type,
            noise_level=noise_level),
        start=start, stop=stop, num=num)

    W.log_x_y(h, noise_levels, brier_score, f"{key}/Brier_Score_vs_noise_level",
              x_label="Noise Level (std dev of noise)",
              y_label="Brier Score")

    noise_levels, entropy = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: Metrics.compute_entropy(
            uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            noise_type=noise_type,
            noise_level=noise_level),
        start=start, stop=stop, num=num)

    W.log_x_y(h, noise_levels, entropy, f"{key}/Entropy_vs_noise_level",
              x_label="Noise Level (std dev of noise)",
              y_label="Entropy")

    noise_levels, kl_dist = UQ.apply_f_to_all_noise_levels(
        function=lambda noise_level: Metrics.compute_kl_divergence_vs_uniform(
            uq_predictor, test_loader,
            limit_batches=h['limit_test_batches'],
            noise_type=noise_type,
            noise_level=noise_level),
        start=start, stop=stop, num=num)

    W.log_x_y(h, noise_levels, kl_dist, f"{key}/KL_Divergence_vs_noise_level",
              x_label="Noise Level (std dev of noise)",
              y_label="KL Divergence")


def ood_experiments_ensemble(h: dict, test_loader: DataLoader, uq_predictor: UQPredictor):
    if h['UQ_predictor'] in ['EnsembleStdBasedSoftmaxRelaxation', 'EnsembleAvgSoftmax']:
        display_predicted_distributions_ood(f"predicted_distributions_ood", test_loader, h, uq_predictor)

        # ECE, accuracy, brier score, entropy under distributional shift
        metrics_under_distributional_shift(f"metrics_under_distributional_shift", test_loader, h, uq_predictor)


def analysis(h: dict, c: ANN, test_loader: DataLoader):
    uq_predictor = UQPredictorFactory.create_instance(
        h, c, predictor_type=h['UQ_predictor'], predictor_params=h['predictor_params'])
    c.eval()

    compute_metrics(h, test_loader, uq_predictor)
    variance_error(f"variance_error", test_loader, h, uq_predictor)
    entropy_ood(f"entropy_ood", test_loader, h, uq_predictor)
    ood_experiments_ensemble(h, test_loader, uq_predictor)
    confidence_vs_avg_accuracy_line(f"confidence_vs_avg_accuracy_line", uq_predictor, test_loader, h)


@init_decorator
@wandb_resume_but_new_run_decorator  # calls wandb.init
@timer_decorator
def main(h: dict):
    train_loader, val_loader, test_loader = data_loaders(h)

    # c: MultiInputClassifier = MultiInputClassifier.create_instance(h)
    c: ANN = ClassifierFactory.create_instance(h)

    c.load()  # load weights
    c.to(h['device'])

    analysis(h, c, test_loader)
    return c


if __name__ == '__main__':
    main(Hyperparameters.get())
    torch.cuda.empty_cache()
