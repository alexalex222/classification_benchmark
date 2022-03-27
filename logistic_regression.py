import os
import gc
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


start_time = None


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.lr(x)


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} megabytes".format(int(torch.cuda.max_memory_allocated()/10e6)))


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    start_timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        data = data.reshape(batch_size, -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    end_timer_and_print('Default precision:')


def train_float16(args, model, device, train_loader, optimizer, criterion, epoch, scaler):
    model.train()
    start_timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        data = data.reshape(batch_size, -1)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            assert output.dtype is torch.float16
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    end_timer_and_print('Mixed precision:')


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            data = data.reshape(batch_size, -1)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mix_precision', action='store_false', help='Use mix precision training')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if os.name == 'nt':
        data_root = 'C:\\Temp\\torch_dataset'
    else:
        data_root = '/media/kuilin/research/temp/torch_dataset'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
    train_loader = DataLoader(
        datasets.CIFAR10(data_root, train=True, download=False,
                       transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        datasets.CIFAR10(data_root, train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = LogisticRegression(input_dim=32*32*3, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.mix_precision:
        print('Use mix precision training.')
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1, args.epochs + 1):
        if args.mix_precision:
            train_float16(args, model, device, train_loader, optimizer, criterion, epoch, scaler)
        else:
            train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader)
        # scheduler.step()

    test(args, model, device, test_loader)

    # if args.save_model:
    #     torch.save(model.state_dict(), "saved_models/cifar100_cnn.pt")


if __name__ == '__main__':
    main()
