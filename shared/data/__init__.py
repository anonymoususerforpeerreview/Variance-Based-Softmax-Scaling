from torch.utils.data import DataLoader

from shared.hyperparameters import Hyperparameters


def filter_iterator(data_loader: DataLoader, limit_batches: float, log_progress: bool = False,
                    log_every_percent: float = 20.0):
    """
    Filter the data loader to only process a certain percentage of the batches + MOVE BATCH TO GPU.
    """
    total_batches = len(data_loader)
    num_batches_to_process = int(total_batches * limit_batches)
    last_logged_percent = -log_every_percent

    for i, batch in enumerate(data_loader):
        if i >= num_batches_to_process:
            break

        # Move items to GPU if available
        device = Hyperparameters.get()['device']
        batch = tuple(item.to(device) for item in batch)  # Move items to GPU

        current_percent = 100 * (i + 1) / num_batches_to_process
        if log_progress and current_percent - last_logged_percent >= log_every_percent:
            print(f"Progress: {current_percent:.2f}%")
            last_logged_percent = current_percent

        yield i, batch


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    data = torch.rand(99, 10)  # 100 samples, 10 features
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=1)

    # Use the filter_iterator function
    for i, batch in filter_iterator(data_loader, limit_batches=0.5, log_progress=True):
        pass
