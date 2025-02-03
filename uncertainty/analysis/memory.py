import torch
import torch.nn as nn
import torch.profiler


# Placeholder model definition
class CifarModelSoftmaxScale(nn.Module):
    def __init__(self):
        super(CifarModelSoftmaxScale, self).__init__()
        self.in_channels = 3
        self.num_classes = 10
        self.tau_amplifier = 1
        self.tau_shift = 0
        self.layers = nn.Sequential(
            # assumes input shape (batch, in_channels, 32, 32)
            nn.Conv2d(self.in_channels, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 32, 32)
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),  # result: (batch, 160, 32, 32)
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),  # result: (batch, 96, 32, 32)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 96, 16, 16)
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 192, 8, 8)
            nn.Dropout(p=0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1, padding=0),  # result: (batch, num_classes, 8, 8)
            # Average pooling will take place in the superclass (_forward_single_with_moving_average)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=8, stride=1, padding=0),  # result: (batch, num_classes, 1, 1)
        )

    def forward(self, x):
        out = self.layers(x)  # (batch, num_classes, 8, 8)
        logits_final = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)

        std = torch.std(out, dim=(2, 3)).mean(dim=1)  # (batch, 1)
        ones = torch.ones_like(std)  # (batch, 1)

        tau = (self.tau_amplifier * std) + self.tau_shift  # (batch, 1)
        tau = torch.max(tau, ones).unsqueeze(-1).expand(-1, self.num_classes)  # (batch, num_classes)

        uncertainty = torch.softmax(logits_final / tau, dim=-1)
        return uncertainty


class CifarModelConventional(nn.Module):
    def __init__(self):
        super(CifarModelConventional, self).__init__()
        self.in_channels = 3
        self.num_classes = 10
        self.layers = nn.Sequential(
            # assumes input shape (batch, in_channels, 32, 32)
            nn.Conv2d(self.in_channels, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 32, 32)
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),  # result: (batch, 160, 32, 32)
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),  # result: (batch, 96, 32, 32)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 96, 16, 16)
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 192, 8, 8)
            nn.Dropout(p=0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1, padding=0),  # result: (batch, num_classes, 8, 8)
            # Average pooling will take place in the superclass (_forward_single_with_moving_average)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=8, stride=1, padding=0),  # result: (batch, num_classes, 1, 1)
        )

    def forward(self, x):
        out = self.layers(x)  # (batch, num_classes, 8, 8)
        logits_final = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)  # (batch, num_classes)
        uncertainty = torch.softmax(logits_final, dim=-1)
        return uncertainty


class CifarModelDropout(nn.Module):
    def __init__(self, num_samples: int):
        super(CifarModelDropout, self).__init__()
        self.in_channels = 3
        self.num_classes = 10
        self.dropout_rate = 0.5
        self.dropout = nn.Dropout(p=0.5)
        self.num_samples = num_samples

        self.layers = nn.ModuleList([
            # assumes input shape (batch, in_channels, 32, 32)
            nn.Conv2d(self.in_channels, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 32, 32)
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),  # result: (batch, 160, 32, 32)
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),  # result: (batch, 96, 32, 32)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 96, 16, 16)
            # nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 192, 8, 8)
            # nn.Dropout(p=0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1, padding=0),  # result: (batch, num_classes, 8, 8)
            # Average pooling will take place in the superclass (_forward_single_with_moving_average)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=8, stride=1, padding=0),  # result: (batch, num_classes, 1, 1)
        ])

    def forward(self, x):
        # create num_samples samples
        x = x.repeat(self.num_samples, 1, 1, 1)  # (num_samples * batch, in_channels, 32, 32)
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = self.dropout(x)  # Apply dropout after each relu

        out = x
        logits_final = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        # (batch, num_classes)

        uncertainty = torch.softmax(logits_final, dim=-1)  # (batch*num_samples, num_classes)
        uncertainty = torch.mean(uncertainty.view(self.num_samples, -1, self.num_classes), dim=0)
        return uncertainty

    def eval(self):
        self.train()  # Set model to training mode ALWAYS (for dropout to remain active)


class CifarModelEnsemble(nn.Module):
    def __init__(self, num_models: int):
        super(CifarModelEnsemble, self).__init__()
        self.in_channels = 3
        self.num_classes = 10
        self.models = nn.ModuleList([CifarModelConventional() for _ in range(num_models)])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        uncertainty = torch.mean(torch.stack(outputs), dim=0)  # Uncertainty estimate

        return uncertainty


# Count FLOPs and memory usage
def compute_flops_and_memory(model, input_tensor):
    # Compute FLOPs for a single forward pass
    # flops = FlopCountAnalysis(model, input_tensor).total()
    flops = 0  # too many operations aren't supported so there's no correct estimate

    # Compute peak memory usage
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,  # Track memory
    ) as prof:
        _ = model(input_tensor)

    memory_usage = max(e.cuda_memory_usage for e in prof.key_averages()) / 1e9  # Peak memory usage
    return flops, memory_usage


def compute_memory():
    memory_usage_dict = {
        "Conventional": [],
        "Custom Method": [],
        "Ensemble (1 model)": [],
        "Ensemble (10 models)": [],
        "MC Dropout (1 sample)": [],
        "MC Dropout (10 samples)": []
    }

    for loops in range(10):
        # clear gpu memory
        device = torch.device("cuda")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Conventional
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        conventional_model = CifarModelConventional().to(device)
        conventional_model.eval()
        flops, memory = compute_flops_and_memory(conventional_model, input_tensor)
        memory_usage_dict["Conventional"].append(memory)
        print(f"Conventional FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Custom Method
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        custom_model = CifarModelSoftmaxScale().to(device)
        custom_model.eval()
        flops, memory = compute_flops_and_memory(custom_model, input_tensor)
        memory_usage_dict["Custom Method"].append(memory)
        print(f"Custom Method FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Ensemble (1 model)
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        num_ensemble_models = 1
        ensemble_model = CifarModelEnsemble(num_models=num_ensemble_models).to(device)
        ensemble_model.eval()
        flops, memory = compute_flops_and_memory(ensemble_model, input_tensor)
        memory_usage_dict["Ensemble (1 model)"].append(memory)
        print(f"Ensemble (1 model) FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Ensemble (10 models)
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        num_ensemble_models = 10
        ensemble_model = CifarModelEnsemble(num_models=num_ensemble_models).to(device)
        ensemble_model.eval()
        flops, memory = compute_flops_and_memory(ensemble_model, input_tensor)
        memory_usage_dict["Ensemble (10 models)"].append(memory)
        print(f"Ensemble (10 models) FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # MC Dropout (1 sample)
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        num_mc_samples = 1
        mc_model = CifarModelDropout(num_samples=num_mc_samples).to(device)
        mc_model.eval()
        flops, memory = compute_flops_and_memory(mc_model, input_tensor)
        memory_usage_dict["MC Dropout (1 sample)"].append(memory)
        print(f"MC Dropout (1 sample) FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # MC Dropout (10 samples)
        input_tensor = torch.randn(128, 3, 32, 32, device=device)  # Batch size 128, 3 channels, 32x32 image
        num_mc_samples = 10
        mc_model = CifarModelDropout(num_samples=num_mc_samples).to(device)
        mc_model.eval()
        flops, memory = compute_flops_and_memory(mc_model, input_tensor)
        memory_usage_dict["MC Dropout (10 samples)"].append(memory)
        print(f"MC Dropout (10 samples) FLOPs: {flops}, Memory Usage: {memory} GB")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return memory_usage_dict


# Main execution
if __name__ == "__main__":
    memory_usage_dict = compute_memory()
    print(memory_usage_dict)
