import argparse
import os
import time
import random
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from sam import SAM
from models import resnet_db


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy)

    def flush(self) -> None:
        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(f"{loss:12.4f}  │{100*accuracy:10.2f} %  ┃", flush=True)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss, accuracy) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")


def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class Cifar:
    def __init__(self, batch_size, threads):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        if os.name == 'nt':
            self.data_dir = 'D:\\Temp\\torch_dataset'
        else:
            self.data_dir = '/media/kuilin/research/temp/torch_dataset'

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root=self.data_dir, train=True,
                                                  download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=self.data_dir, train=False,
                                                 download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.5, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    model = resnet_db.__dict__['resnet12_db'](avg_pool=True, drop_rate=0.1,
                                              dropblock_size=2, num_classes=100).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
