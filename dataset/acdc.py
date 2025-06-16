from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import itertools
from torch.utils.data.sampler import Sampler


class ACDCDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train_l",
        num=None,
        id_path=None,
        nsample=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train_l" or self.split == "train_u":
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if self.split == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]

        elif self.split == "val":
            with open('splits/acdc/valtest.txt', 'r') as f:
            # with open('splits/acdc/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self._base_dir, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        sample = {"image": img, "label": mask}

        if self.split == "train_l" or self.split == "train_u":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = item
        return sample

    def __len__(self):
        return len(self.ids)


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

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


from . import transform as tr

class WeakStrongAugment_CCB(object):
    '''
    image: ndarray of shape (H, W)
    label: ndarray of shape (H, W)
    '''

    def __init__(self, output_size):

        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = self.resize(image)
        label = self.resize(label)
        image_weak = deepcopy(image)
        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(np.array(label)).long()

        ########## weak augmentation is rotation / flip ##########
        image_weak = Image.fromarray((image_weak * 255).astype(np.uint8))
        img_strong = deepcopy(image_weak)
        image_weak = torch.from_numpy(np.array(image_weak)).unsqueeze(0).float() / 255.0

        ########## strong augmentation is color jitter ##########
        if random.random() < 0.8:
            img_strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_strong)
        img_strong = blur(img_strong, p=0.5)
        img_strong = torch.from_numpy(np.array(img_strong)).unsqueeze(0).float() / 255.0

        # img_strong = tr.mocoV2_aug_acdc(img_strong)
        sample = {
            "image": image,
            "label": label,
            "image_weak": image_weak,
            "image_strong": img_strong,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size / x, self.output_size / y), order=0)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size / x, self.output_size / y), order=0)
        label = zoom(label, (self.output_size / x, self.output_size / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # image = torch.from_numpy(image).unsqueeze(0).float()
        # label = torch.from_numpy(np.array(label)).long()
        sample = {"image": image, "label": label}
        return sample

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask


def random_rotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask


import numpy as np


def random_crop(img, mask, crop_size):
    """
    对输入图像和掩码进行随机裁剪。

    参数:
        img (numpy.ndarray): 输入图像，形状为 (H, W, C) 或 (H, W)。
        mask (numpy.ndarray): 输入掩码，形状为 (H, W)。
        crop_size (tuple): 裁剪的目标尺寸，格式为 (crop_height, crop_width)。

    返回:
        cropped_img (numpy.ndarray): 裁剪后的图像。
        cropped_mask (numpy.ndarray): 裁剪后的掩码。
    """
    # 获取图像和掩码的高度和宽度
    h, w = img.shape[:2]
    crop_h, crop_w = crop_size, crop_size

    # 确保裁剪尺寸不超过原始图像尺寸
    if crop_h > h or crop_w > w:
        raise ValueError("裁剪尺寸不能大于原始图像尺寸")

    # 随机生成裁剪的起始位置
    start_h = np.random.randint(0, h - crop_h + 1)
    start_w = np.random.randint(0, w - crop_w + 1)

    # 进行裁剪
    cropped_img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
    cropped_mask = mask[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return cropped_img, cropped_mask

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size / x, self.output_size / y), order=0)

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)
