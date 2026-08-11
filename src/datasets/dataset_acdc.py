import itertools
import os
import random
import re
from glob import glob
from skimage import transform
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None, seq_length=1):
        self._base_dir = base_dir
        self._list_dir = list_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.seq_length = seq_length
        if self._list_dir:
            self.sample_list = self._load_sample_list()
        else:
            self.sample_list = self._build_sample_list_from_directory()

        print("total {} samples".format(len(self.sample_list)))

    def _load_sample_list(self):
        if self.split.find('train') != -1:
            manifest_name = "train.txt"
            data_dir = "ACDC_training_slices"
        elif self.split.find('val') != -1:
            manifest_name = "val.txt"
            data_dir = "ACDC_training_volumes"
        elif self.split.find('test') != -1:
            manifest_name = "test.txt"
            data_dir = "ACDC_training_volumes"
        else:
            raise ValueError(f"Unsupported ACDC split: {self.split}")

        manifest_path = os.path.join(self._list_dir, manifest_name)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Missing ACDC sample manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            sample_list = [line.strip() for line in handle if line.strip()]
        if not sample_list:
            raise ValueError(f"ACDC sample manifest is empty: {manifest_path}")
        if len(sample_list) != len(set(sample_list)):
            raise ValueError(f"ACDC sample manifest contains duplicate entries: {manifest_path}")

        missing = [
            name for name in sample_list
            if not os.path.isfile(os.path.join(self._base_dir, data_dir, name))
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} ACDC files listed in {manifest_path} are missing from "
                f"{os.path.join(self._base_dir, data_dir)}: {preview}"
            )
        return sample_list

    def _build_sample_list_from_directory(self):
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                sample_list.extend(new_data_list)

        else:
            raise ValueError(f"Unsupported ACDC split: {self.split}")
        return sample_list

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            if self.seq_length > 1:
                match = re.match(r'(.+_frame\d+)_slice_(\d+)\.h5', case)
                if match is None:
                    raise ValueError(f"Cannot parse ACDC slice filename for seq_length>1: {case}")
                prefix = match.group(1)
                slice_num = int(match.group(2))
                images, labels = [], []
                for i in range(self.seq_length):
                    curr_idx = slice_num - (self.seq_length - 1) + i
                    curr_idx = max(curr_idx, 0)
                    curr_file = "{}_slice_{}.h5".format(prefix, curr_idx)
                    curr_path = self._base_dir + "/ACDC_training_slices/" + curr_file
                    if not os.path.exists(curr_path):
                        curr_path = self._base_dir + "/ACDC_training_slices/" + case
                    with h5py.File(curr_path, 'r') as h5f:
                        images.append(h5f['image'][:])
                        labels.append(h5f['label'][:])
                image = np.stack(images)
                label = np.stack(labels)
            else:
                h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
                image = h5f['image'][:]
                label = h5f['label'][:]  # fix sup_type to label
                h5f.close()
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    rot_axes = (0, 1) if image.ndim == 2 else (1, 2)
    image = np.rot90(image, k, axes=rot_axes)
    label = np.rot90(label, k, axes=rot_axes)
    axis = np.random.randint(0, 2)
    flip_axis = axis if image.ndim == 2 else axis + 1
    image = np.flip(image, axis=flip_axis).copy()
    label = np.flip(label, axis=flip_axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-30, 30)
    rot_axes = (0, 1) if image.ndim == 2 else (1, 2)
    image = ndimage.rotate(image, angle, axes=rot_axes, order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=rot_axes, order=0, reshape=False)
    return image, label


def random_brightness_contrast(image, label):
    brightness_factor = np.random.uniform(-0.2, 0.2)
    image = image * (1 + brightness_factor)
    contrast_factor = np.random.uniform(0.8, 1.2)
    image = (image - image.mean()) * contrast_factor + image.mean()
    image = np.clip(image, 0, 1) if image.max() <= 1 else np.clip(image, 0, 255)
    return image, label


def random_crop(image, label):
    if image.ndim == 2:
        x, y = image.shape
    else:
        _, x, y = image.shape
    crop_ratio = np.random.uniform(0.8, 1.2)
    crop_size = (int(x * crop_ratio), int(y * crop_ratio))
    if crop_size[0] >= x:
        crop_size = (x, crop_size[1])
    if crop_size[1] >= y:
        crop_size = (crop_size[0], y)
    x_start = np.random.randint(0, x - crop_size[0] + 1)
    y_start = np.random.randint(0, y - crop_size[1] + 1)
    if image.ndim == 2:
        image_crop = image[x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
        label_crop = label[x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
        image = transform.resize(image_crop, (x, y), order=0, preserve_range=True)
        label = transform.resize(label_crop, (x, y), order=0, preserve_range=True)
    else:
        image_crop = image[:, x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
        label_crop = label[:, x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
        imgs, lbls = [], []
        for i in range(image.shape[0]):
            imgs.append(transform.resize(image_crop[i], (x, y), order=0, preserve_range=True))
            lbls.append(transform.resize(label_crop[i], (x, y), order=0, preserve_range=True))
        image = np.stack(imgs)
        label = np.stack(lbls)
    
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.7:
            image, label = random_rotate(image, label)
        if random.random() > 0.7:
            image, label = random_brightness_contrast(image, label)
        
        if random.random() > 0.7:
            image, label = random_crop(image, label)
        if image.ndim == 2:
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            _, x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                imgs, lbls = [], []
                for i in range(image.shape[0]):
                    imgs.append(zoom(image[i], (self.output_size[0] / x, self.output_size[1] / y), order=0))
                    lbls.append(zoom(label[i], (self.output_size[0] / x, self.output_size[1] / y), order=0))
                image = np.stack(imgs)
                label = np.stack(lbls)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)

        assert (image.shape[-2] == self.output_size[0]) and (image.shape[-1] == self.output_size[1])
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
