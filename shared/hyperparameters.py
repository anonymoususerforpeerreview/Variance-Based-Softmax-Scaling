import argparse
import sys
import re
import torch


class Hyperparameters:
    @staticmethod
    def get():
        h = {  # hyperparameters
            'method': 'UQ_rednd',  # UQ_rednd, ensemble, mc_dropout
            'num_models': 1,  # only for ensemble method

            'model_name': None,  # None for default name: 'model.pth', otherwise specify custom names

            # `random_transform`, `identity` (augmentations, splitting into chunks, etc.)
            'redundancy_method': 'random_crop',
            'redundancy_method_params': {
                'crop_size': 100
            },
            # 'redundancy_method_uncertainty_kernel_size': 0,  # 0 for full image, (3,3) for images, 3 for 1d signals

            'dataset': 'MNIST',
            # 'TOY_REGRESSION', 'RADIO', 'LIBRISPEECH', 'LIBRISPEECH_GIM_SUBSET', 'CIFAR10', 'CIFAR100'
            'dataset_path': './data/',
            'remove_first_n_samples': None,

            # UQ_predictor: 'AvgSoftmax', 'StdBasedSoftmaxRelaxation', 'StdBasedSoftmaxRelaxationLearnedParams', 'EnsembleStdBasedSoftmaxRelaxationLearnedParams', 'Naive'
            'UQ_predictor': 'StdBasedSoftmaxRelaxationLearnedParams',
            'predictor_params': {
                'tau_shift': -4.5,  # -4.5 or 'auto', 'mean', 'qXX' for quantile
                'tau_amplifier': 1,
                'kernel_size': 20,
                'std_on_representations': 'both',
                # 'transformation': 'affine' # currenly only softscal ensemble supports this
                # 'logits', 'softmax', 'both', 'logits_relu', 'logits_exp'
            },

            'num_views': 5,

            'loss': 'cross_entropy',  # 'softplus_based_cross_entropy', 'cross_entropy'
            'alpha': 1,  # 1 is max cross_entropy, 0 is max distinct_path_regularization
            # 0.35 seems to kinda work w/ acc 0.9,  # 1 for cross_entropy, 0 for distinct_path_regularization

            'latent_dims': [32, 64],
            'latent_strides': [2, 2],
            'latent_kernel_sizes': [3, 3],
            'latent_padding': [0, 0],
            'final_activation': 'identity',  # identity, sigmoid, tanh, softplus, exponential, relu
            'weight_decay': 0,  # 1e-4

            'log_path': './logs/',
            'epochs': 2,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'num_workers': 4,

            'limit_train_batches': 1.0,
            'limit_val_batches': 1.0,
            'limit_test_batches': 1.0,

            'seed': 49,

            'use_wandb': True,
            'wandb_analysis_independent_of_train_run': False,  # used in analysis scripts

            'wandb_project': '',
            'wandb_entity': '',
            'run_name': '',

            'noise_type': 'gaussian',  # 'gaussian', 'salt_and_pepper', 'poisson', 'speckle'

            'train': True,  # Set this to false if you only want to evaluate the model
            'fast_dev_run': False,
            'overfit_batches': 0.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enable_progress_bar': False,
            'use_only_k_labels': 0
        }
        h = Hyperparameters.overwrite_args_cli(h)
        h['checkpoint_path'] = f"{h['log_path']}/saved_models"

        Hyperparameters.apply_sanity_checks(h)
        return h

    @staticmethod
    def apply_sanity_checks(h):
        if h['use_only_k_labels'] > 0:
            assert h['use_only_k_labels'] == 10 and h['dataset'] == 'LIBRISPEECH_GIM_SUBSET', \
                "Only 10 labels are supported, and only LibriSpeech dataset is supported"

    @staticmethod
    def _adjust_type(val: any) -> any:
        """
        Transform the string value or list of values to the appropriate type (int, float, bool, dict, or str).
        If val is a list, recursively apply type adjustment to each element.
        :param val: The value to adjust
        :return: The value converted to the appropriate type
        """
        if isinstance(val, list):
            # If the input is a list, apply _adjust_type recursively to each element
            return [Hyperparameters._adjust_type(item) for item in val]

        if val == 'True' or val is True:
            return True
        elif val == 'False' or val is False:
            return False
        elif val == 'None':
            return None

        try:
            # Try converting to a number. ('1.0' -> float, '1' -> int)
            # This is important for --limit_train_batches 1.0, where 1.0 is treated as percentage (100% of data)
            # and 1 is treated as 1 batch

            # If the string contains a '.', treat it as a float
            if '.' in str(val):
                return float(val)
            # Otherwise, try converting to int
            return int(val)
        except (ValueError, TypeError):
            # If it can't be converted to a number, check if it's a list or dict
            if isinstance(val, str):
                if val.startswith('[') and val.endswith(']'):
                    try:
                        # Convert string representation of list to actual list
                        list_val = eval(val)
                        if isinstance(list_val, list):
                            return [Hyperparameters._adjust_type(item) for item in list_val]
                    except (SyntaxError, ValueError):
                        pass
                elif val.startswith('{') and val.endswith('}'):  # e.g. {'crop_size':125}
                    try:
                        # Add quotes to keys if they are missing (due to issues with server-side parsing,
                        # so now also support {crop_size:125})
                        # Add quotes to keys if they are missing
                        val = re.sub(r'(\w+):', r'"\1":', val)

                        # add quotes to values if they are missing
                        val = re.sub(r':(\w+)', r':"\1"', val)

                        dict_val = eval(val)
                        if isinstance(dict_val, dict):
                            return {k: Hyperparameters._adjust_type(v) for k, v in dict_val.items()}
                    except (SyntaxError, ValueError):
                        pass
            # If it can't be converted to a number, list, or dict, return as-is
            return val

    @staticmethod
    def overwrite_args_cli(h: dict[str, any]) -> dict[str, any]:
        # Check if the script is being run in a Jupyter notebook
        if 'ipykernel' not in sys.modules:
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            for key, value in h.items():
                parser.add_argument(f'--{key}', type=str, default=value)

            args = parser.parse_args()

            # Overwrite the default hyperparameters with the command-line arguments
            for key, value in vars(args).items():
                if key in h:
                    new_val = Hyperparameters._adjust_type(value)
                    h[key] = new_val

        # e.g.: python main.py --method uq_through_redundancy --dataset MNIST --num_views 5 --alpha 0.1 --train True
        return h
