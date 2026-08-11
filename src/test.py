import argparse
import logging
import os
import random
import sys

if "--no-strict_deterministic" not in sys.argv and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
from utils import test_single_volume
from rwkv_unet import RWKV_UNet
import ttach as tta

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='ACDC', choices=['ACDC'], help='dataset protocol')
parser.add_argument('--root_path', type=str, default=None, help='training data root')
parser.add_argument('--volume_path', type=str, default=None, help='evaluation volume root')
parser.add_argument('--list_dir', type=str, default=None, help='dataset split-list directory')
parser.add_argument('--train_split', type=str, default=None, help='training split name')
parser.add_argument('--val_split', type=str, default=None, help='validation split name')
parser.add_argument('--test_split', type=str, default=None, help='test split name')
parser.add_argument('--eval_split', type=str, default=None, help='evaluation split name')
parser.add_argument('--num_classes', type=int, default=None, help='number of segmentation classes')
parser.add_argument('--seq_length', type=int, default=None, help='adjacent slice sequence length')
parser.add_argument('--z_spacing', type=float, default=None, help='slice spacing used for volume metrics')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action=argparse.BooleanOptionalAction,
                    default=False, help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='prediction output directory')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--max_iterations', type=int, default=None,
                    help='iteration cap used to resolve the default checkpoint path')
parser.add_argument('--num_workers', type=int, default=1, help='evaluation data loader workers')
parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                    default=None, help='enable deterministic cuDNN behavior')
parser.add_argument('--strict_deterministic', action=argparse.BooleanOptionalAction,
                    default=None, help='enable strict deterministic PyTorch execution')
parser.add_argument('--path_specific', type=str, default=None)
args = parser.parse_args()


def _canonical_dataset_name(name):
    normalized = name.strip().lower()
    if normalized == 'acdc':
        return 'ACDC'
    raise ValueError(f"Unsupported dataset: {name}")


def _apply_paper_protocol(args):
    dataset_name = _canonical_dataset_name(args.dataset)
    protocol = {
        'Dataset': ACDC_dataset,
        'volume_path': './data/ACDC',
        'list_dir': None,
        'train_split': 'train',
        'val_split': 'val',
        'test_split': 'test',
        'eval_split': 'test',
        'num_classes': 4,
        'seq_length': 3,
        'z_spacing': 5,
        'strict_deterministic': True,
        'max_iterations': 60000,
    }
    args.dataset = dataset_name
    args.Dataset = protocol['Dataset']
    for key in (
        'volume_path', 'list_dir', 'train_split', 'val_split', 'test_split',
        'eval_split', 'num_classes', 'seq_length', 'z_spacing', 'max_iterations',
    ):
        if getattr(args, key, None) is None:
            setattr(args, key, protocol[key])
    if args.strict_deterministic is None:
        args.strict_deterministic = protocol['strict_deterministic']
    if args.deterministic is None:
        args.deterministic = True
    args.is_pretrain = True
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


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split=args.eval_split, list_dir=args.list_dir,
                           seq_length=args.seq_length)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    valid_count = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        if torch.sum(label) == 0:
            logging.info('idx %d case %s skip_no_gt' % (i_batch, case_name))
            continue
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,
                                      seq_length=args.seq_length)
        metric_list += np.array(metric_i)
        valid_count += 1
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    if valid_count == 0:
        logging.info('no_gt_cases_found')
        return "Testing Finished!"
    metric_list = metric_list / valid_count
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in eval model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



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
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # name the same snapshot defined in train script!
    args.exp = 'rwkv' + dataset_name + str(args.img_size)
    snapshot_path = _build_snapshot_path(args)
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    if args.path_specific != None:
            snapshot=args.path_specific
    net = RWKV_UNet(
        in_channels=1,
        img_size=args.img_size,
        num_classes=args.num_classes,
        pretrained_path=None,
        encoder_drop_path=0.05,
    ).cuda()
    checkpoint = torch.load(snapshot, map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    if isinstance(checkpoint, dict) and checkpoint and next(iter(checkpoint)).startswith('module.'):
        checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}
    load_result = net.load_state_dict(checkpoint, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture does not match the final DMBG-RWKV model. "
            f"missing={load_result.missing_keys[:20]}, unexpected={load_result.unexpected_keys[:20]}"
        )
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
