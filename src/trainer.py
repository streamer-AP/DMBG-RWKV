import argparse
import logging
import os
import random
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
from torch.nn import functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import test_single_volume, BoundaryLoss
from module.boundary_modules import generate_boundary_gt

def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


def make_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs,
            targets.long(),
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def make_seg_loss(args, ignore_index=None):
    if getattr(args, 'seg_loss', 'ce') == 'focal':
        if ignore_index is None:
            return FocalLoss(gamma=2.0)
        return FocalLoss(gamma=2.0, ignore_index=ignore_index)
    if ignore_index is None:
        return CrossEntropyLoss()
    return CrossEntropyLoss(ignore_index=ignore_index)


def trainer_acdc(args, model, snapshot_path):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    seq_length = getattr(args, 'seq_length', 1)
    db_train = BaseDataSets(base_dir=args.root_path, split=args.train_split, list_dir=args.list_dir,
                            transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
                            seq_length=seq_length)
    db_val = BaseDataSets(base_dir=args.volume_path, split=args.val_split, list_dir=args.list_dir)
    worker_init_fn = make_worker_init_fn(args.seed)
    train_generator = make_loader_generator(args.seed)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
                             generator=train_generator)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=args.num_workers)
    max_iterations = args.max_iterations if args.max_iterations > 0 else args.max_epochs * len(trainloader)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr,weight_decay=0.00015)
    scheduler = CosineAnnealingLR(optimizer, T_max=3*args.max_epochs//4, eta_min=0.000001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)
    boundary_loss_fn = BoundaryLoss(num_classes - 1)
    lambda_target = 0.15
    boundary_kernel_size = 3
    warmup_epochs = 10

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))

    iter_num = 0
    best_performance = 0.0
    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if seq_length > 1:
                b, l, c, h, w = volume_batch.shape
                volume_batch = volume_batch.view(b*l, c, h, w)
                label_batch = label_batch.view(b*l, h, w)

            outputs = model(volume_batch, seq_length=seq_length)
            if isinstance(outputs, tuple):
                seg_out, boundary_logits = outputs
            else:
                seg_out, boundary_logits = outputs, None
            loss_ce = ce_loss(seg_out, label_batch[:].long())
            loss_dice = dice_loss(seg_out, label_batch, softmax=True)
            loss = 0.2 * loss_ce + 0.8 * loss_dice
            if boundary_logits is not None:
                boundary_gt = generate_boundary_gt(label_batch, num_classes, kernel_size=boundary_kernel_size)
                loss_b = 0.0
                for bl in boundary_logits:
                    bl_up = F.interpolate(bl, size=label_batch.shape[-2:], mode='bilinear', align_corners=False)
                    loss_b = loss_b + boundary_loss_fn(bl_up, boundary_gt, label_batch, num_classes)
                loss_b = loss_b / len(boundary_logits)
                lambda_b = lambda_target if warmup_epochs <= 0 else lambda_target * min(epoch_num / warmup_epochs, 1.0)
                loss = loss + lambda_b * loss_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    seg_out, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size],
                                                  seq_length=seq_length)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                    iter_num, performance, mean_hd95))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
            scheduler.step()
            if iter_num >= max_iterations:
                iterator.close()
                writer.close()
                return "Training Finished!"
