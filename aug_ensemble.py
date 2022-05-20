# %%
import argparse
import os
import time
import random
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from models import resnet_db


def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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
            torchvision.transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root=self.data_dir, train=True,
                                                  download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=self.data_dir, train=False,
                                                 download=True, transform=train_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
batch_size = 1024
threads = 0
seed = 1
num_models = 8

initialize(seed=seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Cifar(batch_size, threads)

all_model_prediction = torch.zeros((num_models, len(dataset.test.dataset), 100))


# %%
model = resnet_db.__dict__['resnet12_db'](avg_pool=True, drop_rate=0.1,
                                              dropblock_size=2, num_classes=100).to(device)
save_path = './examples/resnet12_{}.pth'.format(1)
model.load_state_dict(torch.load(save_path))

# %%
for i in range(num_models):
    print('Model ', i)
    one_model_prediction = torch.zeros((len(dataset.test.dataset), 100))
    start_index = 0
    model.eval()
    acc_sum = 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            predictions = predictions.cpu()
            one_model_prediction[start_index: start_index + len(targets), :] = predictions
            start_index += len(targets)
            correct = torch.argmax(predictions, 1) == targets.cpu()
            acc_sum += correct.sum().item()

    print('Model {0} acc: {1}'.format(i, acc_sum / len(dataset.test.dataset)))

    all_model_prediction[i] = one_model_prediction

# %%
# ensemble_prediction = torch.mode(torch.argmax(all_model_prediction, -1), 0).values
ensemble_prediction = torch.argmax(all_model_prediction.mean(dim=0), -1)
all_targets = torch.FloatTensor(dataset.test.dataset.targets)
ensemble_correct = ensemble_prediction.float() == all_targets
ensemble_correct = ensemble_correct.double()
print('Ensemble model acc: {0}'.format(ensemble_correct.sum().item() / len(dataset.test.dataset)))

# %%
