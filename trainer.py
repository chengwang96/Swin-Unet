import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from utils import test_single_volume
from glob import glob

from sklearn.model_selection import train_test_split

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose, OneOf


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def trainer(args, model, snapshot_path):
    from datasets.dataset_synapse import UNextDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    data_dir = 'data'
    dataset_name = args.dataset
    img_ext = '.png'
    if dataset_name == 'chase':
        img_ext = '.jpg'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'chase':
        mask_ext = '_1stHO.png'

    dataseed = args.dataseed
    print('dataseed = ' + str(dataseed))
    input_h = args.input_size
    input_w = args.input_size
    print('input_size = ' + str(args.input_size))

    img_ids = sorted(
        glob(os.path.join(data_dir, dataset_name, 'images', '*' + img_ext))
    )
    img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=dataseed)

    train_transform = Compose([
        RandomRotate90(),
        # transforms.Flip(),
        geometric.transforms.Flip(),
        Resize(input_h, input_w),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(input_h, input_w),
        transforms.Normalize(),
    ])

    train_dataset = UNextDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(data_dir, dataset_name, 'images'),
        mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=num_classes,
        transform=train_transform)
    val_dataset = UNextDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(data_dir ,dataset_name, 'images'),
        mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=num_classes,
        transform=val_transform)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, 1, shuffle=False)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(dataloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(dataloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(dataloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.squeeze().contiguous().long())
            loss_dice = dice_loss(outputs[:, 1, :, :].unsqueeze(dim=1), label_batch.squeeze(), softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        save_interval = 10  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            iou_avg_meter = AverageMeter()
            dice_avg_meter = AverageMeter()
            hd95_avg_meter = AverageMeter()

            for i_batch, sampled_batch in enumerate(valloader):
                image_batch, label_batch = sampled_batch[0], sampled_batch[1]
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)

                iou, dice, hd95_ = iou_score(outputs[:, 1, :, :].unsqueeze(dim=1), label_batch)
                iou_avg_meter.update(iou, image_batch.size(0))
                dice_avg_meter.update(dice, image_batch.size(0))
                hd95_avg_meter.update(hd95_, image_batch.size(0))

            print('epoch {}'.format(epoch_num))
            print('IoU: %.4f' % iou_avg_meter.avg)
            print('Dice: %.4f' % dice_avg_meter.avg)
            print('HD95: %.4f' % hd95_avg_meter.avg)

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"