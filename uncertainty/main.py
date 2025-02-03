from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from shared.data.data_loader import data_loaders
from shared.decorators import init_decorator, wandb_decorator, timer_decorator
from shared.hyperparameters import Hyperparameters
from uncertainty.uq_through_redundancy.classifier_factory import ClassifierFactory
from uncertainty.uq_through_redundancy.mc_dropout_classifier import MCDropoutClassifier
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier


def train(h: dict, c: MultiInputClassifier, trainer: Trainer, train_loader, val_loader):
    if h['train']:
        try:
            trainer.fit(c, train_loader, val_loader)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
        finally:
            c.save()

    return c.load()


def analysis(h: dict, c: MultiInputClassifier, trainer: Trainer, test_loader: DataLoader):
    """Short performance analysis based on the full inputs etc. real UQ analysis is in `analysis.py`"""

    c.eval()
    # Evaluate the model & print accuracy
    results = trainer.test(c, test_loader)
    print(f"Test Accuracy: {results[0]['test_acc']:.4f}")


@init_decorator
@wandb_decorator  # calls wandb.init
@timer_decorator
def main(h: dict):
    train_loader, val_loader, test_loader = data_loaders(h)

    # assert h['method'] == 'UQ_rednd', "Training is only supported for UQ_rednd method."
    # ^^ that's true, but we still want `method` == 'ensemble' such that logs under correct wandb project name
    # to train an ensemble, run main.py multiple times with different seeds and `num_models` > 1 and `model_name` = model_{i}
    # c: MultiInputClassifier = MultiInputClassifier.create_instance(h)
    c: MultiInputClassifier = ClassifierFactory.create_instance_train(h)

    trainer = Trainer(max_epochs=h['epochs'], fast_dev_run=h['fast_dev_run'], overfit_batches=h['overfit_batches'],
                      devices="auto", strategy="auto",
                      limit_train_batches=h['limit_train_batches'],
                      limit_val_batches=h['limit_val_batches'],
                      limit_test_batches=h['limit_test_batches'],
                      logger=WandbLogger() if h['use_wandb'] else None,
                      enable_progress_bar=h['enable_progress_bar'])

    train(h, c, trainer, train_loader, val_loader)

    # Short performance analysis based on the full inputs etc. real UQ analysis is in `analysis.py`
    analysis(h, c, trainer, test_loader)
    return c


if __name__ == "__main__":
    main(Hyperparameters.get())
