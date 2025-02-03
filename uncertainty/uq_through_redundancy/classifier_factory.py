from shared.abstract_model import ANN
from uncertainty.uq_through_redundancy.cifar_classifier import NetworkInNetworkClassifier, \
    MCDropoutNetworkInNetworkClassifier
from uncertainty.uq_through_redundancy.ensemble_classifier import EnsembleClassifier
from uncertainty.uq_through_redundancy.mc_dropout_classifier import MCDropoutClassifier
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier
from shared.data.dataset_meta import DatasetMeta as M


class ClassifierFactory:

    @staticmethod
    def create_instance(h: dict) -> ANN:
        """
        Create an instance of a classifier based on the hyperparameters.
        :param h: The hyperparameters
        :return: The classifier instance
        """
        if h['method'] == 'ensemble':
            return EnsembleClassifier.create_instance(h)
        elif h['method'] == 'UQ_rednd':
            if M.is_vision_dataset(h): # CIFAR10, CIFAR100, MNIST
                return NetworkInNetworkClassifier.create_instance(h)
            else: # Radio, LibriSpeech, LibriSpeech_GIM_SUBSET
                return MultiInputClassifier.create_instance(h)
        elif h['method'] == 'mc_dropout':
            if M.is_vision_dataset(h):
                return MCDropoutNetworkInNetworkClassifier.create_instance(h)
            else:
                return MCDropoutClassifier.create_instance(h)
        else:
            raise ValueError(f"Model {h['model']} not supported.")

    @staticmethod
    def create_instance_train(h: dict) -> MultiInputClassifier:
        # Ensemble and UQ_rednd are the same as in create_instance
        if h['method'] in ['ensemble', 'UQ_rednd']:
            if M.is_vision_dataset(h):
                return NetworkInNetworkClassifier.create_instance(h)
            else:
                return MultiInputClassifier.create_instance(h)
        elif h['method'] == 'mc_dropout':
            if M.is_vision_dataset(h):
                return MCDropoutNetworkInNetworkClassifier.create_instance(h)
            else:
                return MCDropoutClassifier.create_instance(h)
        else:
            raise ValueError(f"Model {h['model']} not supported.")
