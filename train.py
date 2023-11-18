#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
from pathlib import Path
from dataset import MagnaTagATune
from evaluation import evaluate
import os
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(
    description="Train a simple CNN on Magnatagatune dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
#path = '/mnt/storage/scratch/ge20118/MagnaTagATune/'
cwd = os.getcwd()
path = os.path.join(cwd, 'MagnaTagATune', 'MagnaTagATune')
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-1, type=float, help="Learning rate")
parser.add_argument("--length", default=256, type=int, help="length")
parser.add_argument("--stride", default=256, type=int, help="stride")
parser.add_argument(
    "--batch-size",
    default=2,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print('cuda')
else:
    DEVICE = torch.device("cpu")
    print('cpu')

def main(args):
    # print(args.worker_count)
    #transform = transforms.ToTensor()
    #args.dataset_root.mkdir(parents=True, exist_ok=True)
    # train_dataset = MagnaTagATune(dataset_path=f'{path}annotations/train_labels.pkl', samples_path=f'{path}samples/')
    # test_dataset = MagnaTagATune(dataset_path=f'{path}annotations/val_labels.pkl', samples_path=f'{path}samples/')
    train_dataset = MagnaTagATune(dataset_path=os.path.join(path, 'annotations', 'train_labels.pkl'), samples_path=os.path.join(path, 'samples'))
    test_dataset = MagnaTagATune(dataset_path=os.path.join(path, 'annotations', 'val_labels.pkl'), samples_path=os.path.join(path, 'samples'))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=10,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
    )
    model = CNN(args=args)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # TODO in channels?
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=args.length, stride=args.stride)
        self.initialise_layer(self.conv1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.initialise_layer(self.conv2)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.initialise_layer(self.conv3)
        self.fc1 = nn.Linear(32*6, 100)
        self.initialise_layer(self.fc1)
        # TODO is this right?
        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

    def forward(self, input: torch.Tensor):
        print(f'before first flatten {input.size()}')
        x = torch.flatten(input, 0, 1)
        print(f'after first flatten {x.size()}')
        x = F.relu(self.conv1(x))
        print(f'after conv1 {x.size()}')
        x = F.relu(self.conv2(x))
        print(f'after conv2 {x.size()}')
        x = self.pool(x)
        print(f'after pool1 {x.size()}')
        x = F.relu(self.conv3(x))
        print(f'after conv3 {x.size()}')
        x = self.pool(x)
        print(f'after pool2 {x.size()}')
        x = torch.flatten(x, start_dim=1)
        print(f'after flatten2 {x.size()}')
        x = F.relu(self.fc1(x))
        # TODO sigmoid fc2 or fc3?
        x = F.sigmoid(self.fc2(x))
        #print(f'after sigmoid', x)
        x = x.reshape(input.shape[0], input.shape[1], 50)
        x = torch.mean(x, 1)
        print(f'after mean {x.size()}')
        #x = x.reshape(-1, 1)
        print(f'after reshape {x.size()}')
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for filename, batch, labels in self.train_loader:
                print(len(batch))
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                
                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    preds = logits.argmax(-1).reshape(-1, 1)
                    # print("preds", preds.shape)
                    #accuracy = compute_accuracy(labels, preds)
                    #accuracy = evaluate(preds=preds, gts_path=f'{path}annotations/train.pkl')
                    accuracy = evaluate(preds=logits, gts_path=os.path.join(path, 'annotations', 'train_labels.pkl'))

                    
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)
                    
                self.step += 1
                data_load_start_time = time.time()
        
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
    
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu()
                results["preds"].extend(list(preds))
               # results["labels"].extend(list(labels.cpu().numpy()))

        # accuracy = compute_accuracy(
        #     np.array(results["labels"]), np.array(results["preds"])
        # )

        accuracy = evaluate(preds=preds, gts_path=f'{path}annotations/val.pkl')
        
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())