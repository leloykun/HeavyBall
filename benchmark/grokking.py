import random
import heavyball
import itertools
from typing import List
from collections import defaultdict
import copy
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from heavyball.utils import set_torch
from benchmark.utils import get_optim

GROKKING_THRESHOLD = 0.9

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class ModularMLP(nn.Module):
    def __init__(self, numbers: int, p: int, hidden_dim: int, bias: bool = True, rms_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(p, hidden_dim),
            nn.RMSNorm(hidden_dim, elementwise_affine=False) if rms_norm else nn.Identity(),
            nn.Flatten(),
            nn.Linear(numbers * hidden_dim, hidden_dim, bias=bias),
            nn.LeakyReLU(),
            nn.RMSNorm(hidden_dim, elementwise_affine=False) if rms_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            nn.LeakyReLU(),
            nn.RMSNorm(hidden_dim, elementwise_affine=False) if rms_norm else nn.Identity(),
            nn.Linear(hidden_dim, p, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ModuloDataset(torch.utils.data.Dataset):
    def __init__(self, p, numbers, min_idx, length, batch_size):
        length = length // batch_size
        self.p = p
        self.numbers = numbers
        self.n_samples = length
        self.min_idx = min_idx
        self.max_idx = min_idx + length
        self.batch_size = batch_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(random.Random(min(idx + self.min_idx, self.max_idx)).randint(0, 2**32))
        x = torch.randint(0, self.p, (self.batch_size, self.numbers), generator=generator)
        y = (x.sum(dim=-1) % self.p).long()
        return x, y


def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().detach()
            total += y.size(0)
    return correct / total


def plot_results(train_losses, test_accs, steps_to_grok=None, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)

    # Plot test accuracy
    eval_steps = np.arange(0, len(train_losses)+1, len(train_losses) // (len(test_accs) - 1))
    ax2.plot(eval_steps, test_accs, label='Test Accuracy', color='orange')
    ax2.axhline(y=GROKKING_THRESHOLD, color='r', linestyle='--', label='Grokking Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_title('Test Accuracy Over Time')
    ax2.grid(True)

    if steps_to_grok is not None:
        ax2.axvline(x=steps_to_grok, color='g', linestyle='--', 
                   label=f'Grokking Step ({steps_to_grok})')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_results_muon(muon_scale_mode_train_loss_map, muon_scale_mode_test_acc_map, muon_scale_mode_steps_to_grok_map, save_path=None):
    assert len(muon_scale_mode_train_loss_map) == len(muon_scale_mode_test_acc_map) == len(muon_scale_mode_steps_to_grok_map)
    scale_modes = list(muon_scale_mode_train_loss_map.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1: plt.Axes = ax1
    ax2: plt.Axes = ax2
    color_palette = plt.cm.get_cmap('tab10', len(scale_modes))

    # Plot training loss
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)
    for idx, curr_scale_mode in enumerate(scale_modes):
        ax1.plot(
            muon_scale_mode_train_loss_map[curr_scale_mode],
            label=f'Training Loss; scale_mode={curr_scale_mode}',
            color=color_palette(idx),
        )

    # Plot test accuracy
    ax2.axhline(y=GROKKING_THRESHOLD, color='r', linestyle='--', label='Grokking Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_title('Test Accuracy Over Time')
    ax2.grid(True)
    for idx, curr_scale_mode in enumerate(scale_modes):
        train_losses = muon_scale_mode_train_loss_map[curr_scale_mode]
        test_accs = muon_scale_mode_test_acc_map[curr_scale_mode]
        steps_to_grok = muon_scale_mode_steps_to_grok_map[curr_scale_mode]
        eval_steps = np.arange(0, len(train_losses)+1, len(train_losses) // (len(test_accs) - 1))
        ax2.plot(eval_steps, test_accs, label=f'Test Accuracy; scale_mode={curr_scale_mode}', color=color_palette(idx))
        if steps_to_grok is not None:
            ax2.axvline(x=steps_to_grok, color=color_palette(idx), linestyle='--')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'),
         opt: List[str] = typer.Option(['AdamW', 'OrthoAdamW', 'AdamWOrtho', 'ForeachPSGDKron', 'Muon'], help='Optimizers to use'),
         steps: int = 2000,
         batch_size: int = 32,
         hidden_dim: int = 32,
         p: int = 257,
         numbers: int = 4,
         weight_decay: float = 0,
         lr: float = 1e-3,
         train_percent: float = 0.1,
         eval_samples: int = 1024,
         printervall: int = 1000,
         scale_mode: List[str] = typer.Option(["none", "none", "scale", "dual_norm", "modular_norm"]),
         bias: bool = True,
         rms_norm: bool = False,
         seed: int = 42):

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dtype = [getattr(torch, curr_dtype) for curr_dtype in dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_dir = Path('.')
    # Clean up old plots
    for path in plot_dir.glob('grokking_*.png'):
        path.unlink()

    # Pre-generate datasets
    unique_samples = p ** numbers
    train_data = ModuloDataset(p, numbers, 0, int(unique_samples * train_percent), batch_size)
    test_data = ModuloDataset(p, numbers, train_data.max_idx, eval_samples, eval_samples)

    print(f"Training on {train_data.n_samples * batch_size:,} samples - {train_percent * 100}%")
    print(f"Testing on {eval_samples:,} samples")


    train_loader = DataLoader(
        train_data, 
        collate_fn=lambda x: x[0],
        batch_size=1, 
        shuffle=False,
        pin_memory=True,    
        num_workers=4,
        drop_last=True,
        prefetch_factor=16,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_data, 
        collate_fn=lambda x: x[0],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        prefetch_factor=32
    )
    test_loader = list(test_loader)
    test_loader = [[x.pin_memory() for x in i] for i in test_loader]

    criterion = nn.CrossEntropyLoss()

    global_model = ModularMLP(numbers, p, hidden_dim, bias, rms_norm).to(device)
    global_model = torch.compile(global_model, mode='max-autotune-no-cudagraphs')
    muon_dtype_scale_mode_train_loss_map = {}
    muon_dtype_scale_mode_test_acc_map = {}
    muon_dtype_scale_mode_steps_to_grok_map = {}
    for curr_dtype, curr_opt, curr_scale_mode in itertools.product(dtype, opt, scale_mode):
        train_iter = iter(train_loader)
        history = defaultdict(list)
        def data():
            """Get next batch from the dataloader"""
            nonlocal train_iter
            try:
                x, y = next(train_iter)
            except (StopIteration, NameError):
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            return x.to(device), y.to(device)

        if curr_opt == "Muon":
            print(f"\nRunning {curr_opt} with {curr_dtype} & scale_mode={curr_scale_mode}")
        else:
            if curr_scale_mode != 'none':
                continue
            print(f"\nRunning {curr_opt} with {curr_dtype}")
        model = copy.deepcopy(global_model)
        model.to(dtype=curr_dtype)

        # Get optimizer class
        optimizer_class = getattr(heavyball, curr_opt)
        if curr_opt == "Muon":
            muon_params = []
            adam_params = []
            for param in model.parameters():
                if param.ndim == 2 and not (param.shape[0] == p or param.shape[1] == p):
                    muon_params.append(param)
                    print("Muon", param.ndim, param.shape)
                else:
                    adam_params.append(param)
                    print("Adam", param.ndim, param.shape)
            optimizers = [
                get_optim(optimizer_class, muon_params, lr=lr, weight_decay=weight_decay, scale_mode=curr_scale_mode),
                get_optim(heavyball.AdamW, adam_params, lr=lr, weight_decay=weight_decay),
            ]
        else:
            optimizers = [get_optim(optimizer_class, model.parameters(), lr=lr, weight_decay=weight_decay)]

        loss_hist = torch.empty(steps+1)

        # Training loop
        for step in range(steps+1):
            model.train()
            x, y = data()

            model.zero_grad(set_to_none=True)
            # optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            with torch.no_grad():
                loss_hist[step] = loss.detach()

                if step % printervall == 0:
                    lh = loss_hist[:step][-printervall:].mean().item()
                    acc = evaluate(model, test_loader, device).item()
                    history['test_acc'].append(acc)
                    print(f"Step {step}: Loss = {lh:.4f}, Test Acc = {acc:.4f}")

        # Plot results
        if curr_opt == "Muon":
            plot_name = plot_dir / f"grokking_{curr_opt}_{curr_dtype}_lr{lr}_h{hidden_dim}_p{p}_scale={curr_scale_mode}_{bias=}_rmsnorm={rms_norm}.png"
        else:
            plot_name = plot_dir / f"grokking_{curr_opt}_{curr_dtype}_lr{lr}_h{hidden_dim}_p{p}_{bias=}_rmsnorm={rms_norm}.png"
        steps_to_grok = next((i*printervall for i, acc in enumerate(history['test_acc']) if acc > GROKKING_THRESHOLD), None)
        plot_results(
            loss_hist.cpu().numpy(),
            history['test_acc'],
            steps_to_grok,
            plot_name,
        )
        print(f"Training curves saved to {plot_name}")

        if curr_opt == "Muon":
            muon_dtype_scale_mode_train_loss_map[(curr_dtype, curr_scale_mode)] = loss_hist.cpu().numpy()
            muon_dtype_scale_mode_test_acc_map[(curr_dtype, curr_scale_mode)] = history['test_acc']
            muon_dtype_scale_mode_steps_to_grok_map[(curr_dtype, curr_scale_mode)] = steps_to_grok

    if "Muon" in opt:
        if isinstance(dtype, str):
            dtype = [dtype]
        for curr_dtype in dtype:
            muon_scale_mode_train_loss_map = {}
            muon_scale_mode_test_acc_map = {}
            muon_scale_mode_steps_to_grok_map = {}
            for curr_scale_mode in scale_mode:
                muon_scale_mode_train_loss_map[curr_scale_mode] = muon_dtype_scale_mode_train_loss_map[(curr_dtype, curr_scale_mode)]
                muon_scale_mode_test_acc_map[curr_scale_mode] = muon_dtype_scale_mode_test_acc_map[(curr_dtype, curr_scale_mode)]
                muon_scale_mode_steps_to_grok_map[curr_scale_mode] = muon_dtype_scale_mode_steps_to_grok_map[(curr_dtype, curr_scale_mode)]
            plot_name = plot_dir / f"grokking_Muon_{curr_dtype}_lr{lr}_h{hidden_dim}_p{p}_scale=all_{bias=}_rmsnorm={rms_norm}.png"
            plot_results_muon(muon_scale_mode_train_loss_map, muon_scale_mode_test_acc_map, muon_scale_mode_steps_to_grok_map, plot_name)
            print(f"Comparative test acc curves for Muon saved to {plot_name}")

if __name__ == '__main__':
    app()
