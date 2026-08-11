import argparse
import logging
import os
import random
import sys
from pathlib import Path

if "--no-strict_deterministic" not in sys.argv and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from rwkv_unet import RWKV_UNet
from trainer import trainer_acdc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_PATH = PROJECT_ROOT / "checkpoints" / "net_B.pth"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='ACDC', choices=['ACDC'], help='dataset protocol')
parser.add_argument('--root_path', type=str, default=None, help='training data root')
parser.add_argument('--volume_path', type=str, default=None, help='validation/test volume root')
parser.add_argument('--list_dir', type=str, default=None, help='dataset split-list directory')
parser.add_argument('--train_split', type=str, default=None, help='training split name')
parser.add_argument('--val_split', type=str, default=None, help='validation split name')
parser.add_argument('--test_split', type=str, default=None, help='test split name')
parser.add_argument('--num_classes', type=int, default=None, help='number of segmentation classes')
parser.add_argument('--seq_length', type=int, default=None, help='adjacent slice sequence length')
parser.add_argument('--z_spacing', type=float, default=None, help='slice spacing used for volume metrics')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--max_iterations', type=int, default=None,
                    help='iteration cap; use 0 to derive it from epochs')
parser.add_argument('--num_workers', type=int, default=8, help='data loader workers')
parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                    default=None, help='enable deterministic cuDNN behavior')
parser.add_argument('--strict_deterministic', action=argparse.BooleanOptionalAction,
                    default=None, help='enable strict deterministic PyTorch execution')
parser.add_argument('--pretrained_path', type=str,
                    default=None,
                    help='optional encoder pretraining checkpoint')
args = parser.parse_args()


def _canonical_dataset_name(name):
    normalized = name.strip().lower()
    if normalized == 'acdc':
        return 'ACDC'
    raise ValueError(f"Unsupported dataset: {name}")


def _apply_paper_protocol(args):
    dataset_name = _canonical_dataset_name(args.dataset)
    protocol = {
        'root_path': './data/ACDC',
        'list_dir': None,
        'volume_path': './data/ACDC',
        'train_split': 'train',
        'val_split': 'val',
        'test_split': 'test',
        'num_classes': 4,
        'seq_length': 3,
        'z_spacing': 5,
        'seg_loss': 'ce',
        'strict_deterministic': True,
        'max_iterations': 60000,
    }
    args.dataset = dataset_name
    for key in (
        'root_path', 'list_dir', 'volume_path', 'train_split', 'val_split',
        'test_split', 'num_classes', 'seq_length', 'z_spacing', 'max_iterations',
    ):
        if getattr(args, key, None) is None:
            setattr(args, key, protocol[key])
    args.seg_loss = protocol['seg_loss']
    if args.strict_deterministic is None:
        args.strict_deterministic = protocol['strict_deterministic']
    if args.deterministic is None:
        args.deterministic = True
    args.n_gpu = 1
    args.is_pretrain = True
    if args.pretrained_path is None:
        args.pretrained_path = str(DEFAULT_PRETRAINED_PATH)
    return dataset_name


def _build_snapshot_path(args):
    snapshot_path = "./outputs/{}/{}".format(args.exp, 'exp1')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    if args.max_iterations > 0 and args.max_iterations != 60000:
        iter_tag = f"{args.max_iterations // 1000}k" if args.max_iterations % 1000 == 0 else str(args.max_iterations)
        snapshot_path = snapshot_path + '_' + iter_tag
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    return snapshot_path


if __name__ == "__main__":
    dataset_name = _apply_paper_protocol(args)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.strict_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.pretrained_path and not Path(args.pretrained_path).exists():
        raise FileNotFoundError(
            f"Missing encoder pretraining checkpoint: {args.pretrained_path}. "
            "Download net_B.pth following README.md, or pass --pretrained_path."
        )
    args.exp = 'rwkv' + dataset_name + str(args.img_size)
    snapshot_path = _build_snapshot_path(args)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    net = RWKV_UNet(
        in_channels=1,
        img_size=args.img_size,
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path,
        encoder_drop_path=0.05,
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)

    net = net.to('cuda')
    trainer_acdc(args, net, snapshot_path)
