import time
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet

from models import resnet_db, lambda_net
from image_transformation import aug_transform, standard_transform
from image_folder_dataset import ImageFolderDataset
from utils import parse_option_pretrain, train_mix_precision, validate, adjust_learning_rate


def main():

    opt = parse_option_pretrain()

    if opt.dataset == 'CUB':
        base_class_num = 200
        image_size = 224
    elif opt.dataset == 'miniImageNet':
        base_class_num = 100
        image_size = 224
    else:
        raise ValueError('Undefined dataset!')

    train_dataset = ImageFolderDataset(
        root=opt.data_root,
        transform=aug_transform,
        num_select_classes=base_class_num,
        num_base_classes=base_class_num,
        train_val_test=0)

    val_dataset = ImageFolderDataset(
        root=opt.data_root,
        transform=standard_transform,
        num_select_classes=base_class_num,
        num_base_classes=base_class_num,
        train_val_test=1)

    test_dataset = ImageFolderDataset(
        root=opt.data_root,
        transform=standard_transform,
        num_select_classes=base_class_num,
        num_base_classes=base_class_num,
        train_val_test=2)

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    if opt.model.endswith('db'):
        model = resnet_db.__dict__[opt.model](avg_pool=True, drop_rate=0.1, dropblock_size=7, num_classes=base_class_num)
    elif opt.model.startswith('lambda'):
        model = lambda_net.__dict__[opt.model](num_classes=base_class_num)
    else:
        model = resnet.__dict__[opt.model](num_classes=base_class_num)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_folder)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    scaler = GradScaler()

    # routine: supervised pre-training
    best_acc = 0.0
    for epoch in range(1, opt.epochs + 1):
        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train_mix_precision(epoch, train_loader, model, criterion, optimizer, scaler, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion)

        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('test_acc_top5', test_acc_top5, epoch)
        logger.add_scalar('test_loss', test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'opt': opt,
                'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            torch.save(state, save_file)

    checkpoint_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint found!'
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    test_acc, test_acc_top5, test_loss = validate(test_loader, model, criterion)

    print('==============================================')
    print('Validation accuracy: ', best_acc)
    print('Test accuracy: ', test_acc)


if __name__ == '__main__':
    main()
