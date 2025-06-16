import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from scipy import ndimage


# # # # # # # # # # # # # # # # # # # # # # # #
# # # my aug
# # # # # # # # # # # # # # # # # # # # # # # #
normalize_moco = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

class moco_GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def mocoV1_aug(image, size=256):
    '''
        使用moco_v1作为强增强
    '''
    image = transforms.Compose([
        # transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_moco,
    ])(image)

    return image


def mocoV2_aug(image, size=256):
    '''
    使用moco_v2作为强增强
    '''
    image = transforms.Compose([
        # transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco_GaussianBlur([0.1, 2.0])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize_moco,
    ])(image)

    return image


# These augmentations are defined exactly as proposed in the paper
def global_augment_DINO(images):
    images = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Larger crops
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(images)
    return images

def global_augment_DINO_org(images):
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Larger crops
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([global_transform(img) for img in images])

def multiple_local_augment_DINO_org(images, num_crops=6):
    size = 96  # Smaller crops for local
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.05, 0.4)),  # Smaller, more concentrated crops
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Color jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([global_transform(img) for img in images])

# # # # # # # # # # # # # # # # # # # # # # # #
# # # SSL4MIS-master原始
# # # # # # # # # # # # # # # # # # # # # # # #
def random_rot_flip(image, label=None):
    # 随机旋转
    k = np.random.randint(0, 4)
    image = image.rotate(90 * k)
    if label is not None:
        label = label.rotate(90 * k)
    # 随机翻转
    axis = np.random.randint(0, 2)
    if axis == 0: # 水平
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if label is not None:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            return image, label
        else:
            return image
    elif axis == 1: #垂直
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if label is not None:
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
            return image, label
        else:
            return image

def random_rot_flip_org(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    # 生成一个随机角度（在-20到20之间）
    angle = random.randint(-20, 20)

    # 旋转图像和标签
    rotated_image = image.rotate(angle)
    rotated_label = label.rotate(angle)

    # 返回旋转后的PIL图像对象
    return rotated_image, rotated_label


def random_rotate_org(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(pil_image):
    # 将 PIL 图像转换为 PyTorch 张量
    tensor_to_pil = transforms.ToPILImage()
    tensor_to_np = transforms.ToTensor()
    # 将 PIL 图像转换为张量
    image_tensor = tensor_to_np(pil_image)
    # s 是颜色抖动的强度
    s = 1.0
    # 创建 ColorJitter 转换对象
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    # 应用颜色抖动
    jittered_image_tensor = jitter(image_tensor)
    # 将抖动后的张量转换回 PIL 图像
    jittered_pil_image = tensor_to_pil(jittered_image_tensor)

    # 返回 PIL 图像
    return jittered_pil_image


def color_jitter_org(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

# # # # # # # # # # # # # # # # # # # # # # # #
# # # 新增
# # # # # # # # # # # # # # # # # # # # # # # #

def adjust_brightness(image, factor_range=(0.5, 1.5)):
    """
    随机调整图像的亮度
    :param image: PIL图像
    :param factor_range: 亮度调整因子的范围（默认0.5到1.5）
    :return: 调整亮度后的图像
    """
    factor = random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

from PIL import Image, ImageDraw

def add_highlight(image, intensity_range=(100, 255), size_range=(10, 50), num_highlights=3):
    """
    在图像中添加随机的高光区域（如白色斑点或渐变）
    :param image: PIL图像
    :param intensity_range: 高光强度范围（默认100到255）
    :param size_range: 高光区域大小范围（默认10到50像素）
    :param num_highlights: 高光区域数量（默认3个）
    :return: 添加高光后的图像
    """
    image = image.convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(num_highlights):
        # 随机生成高光区域的位置和大小
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(size_range[0], size_range[1])
        intensity = random.randint(intensity_range[0], intensity_range[1])

        # 绘制高光区域（圆形）
        draw.ellipse([x, y, x + size, y + size], fill=(intensity, intensity, intensity, 128))

    # 将高光区域叠加到原图上
    return Image.alpha_composite(image, overlay).convert("RGB")

def histogram_equalization(image):
    """
    对图像进行直方图均衡化
    :param image: PIL图像
    :return: 直方图均衡化后的图像
    """
    return ImageOps.equalize(image)


def add_noise(image, noise_intensity=25):
    """
    在图像中添加随机噪声
    :param image: PIL图像
    :param noise_intensity: 噪声强度（默认25）
    :return: 添加噪声后的图像
    """
    image_np = np.array(image)
    noise = np.random.randint(-noise_intensity, noise_intensity, image_np.shape, dtype=np.int32)
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


# # # # # # # # # # # # # # # # # # # # # # # #
# # # unimatch
# # # # # # # # # # # # # # # # # # # # # # # #
def crop(imgA, imgB, mask, size, ignore_value=255):
    w, h = imgA.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    imgA = ImageOps.expand(imgA, border=(0, 0, padw, padh), fill=0)
    imgB = ImageOps.expand(imgB, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = imgA.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    imgA = imgA.crop((x, y, x + size, y + size))
    imgB = imgB.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return imgA, imgB, mask

def crop1(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def hflip(imgA, imgB, mask, p=0.5):
    if random.random() < p:
        imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
        imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return imgA, imgB, mask

def hflip1(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(imgA, imgB, mask, ratio_range):
    w, h = imgA.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    imgA = imgA.resize((ow, oh), Image.BILINEAR)
    imgB = imgB.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return imgA, imgB, mask

def resize1(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

# # # # # # # # # # # # # # # # # # # # # # # #
# # # AugSeg
# # # # # # # # # # # # # # # # # # # # # # # #
class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std)
        assert len(mean) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label):
        in_image = Image.fromarray(np.uint8(in_image))
        image = self.normalize(self.to_tensor(in_image))
        # image = self.to_tensor(in_image)
        label = torch.from_numpy(np.array(in_label, dtype=np.int32)).long()

        return image, label

def build_additional_strong_transform(num_augs=3, flag_use_rand_num=True):

    strong_aug_nums = num_augs
    flag_use_rand_num = flag_use_rand_num
    strong_aug = strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_aug

def build_basic_transfrom(mean=[0.485, 0.456, 0.406]):
    ignore_label = 255
    trs_form = []

    trs_form.append(Resize(500, [0.5, 2.0]))

    trs_form.append(RandomFlip(prob=0.5, flag_hflip=True))

    # crop also sometime for validating
    crop_size, crop_type = [256, 256], 'rand'
    trs_form.append(Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))

    return Compose(trs_form)

class Compose(object):
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label):
        for idx, t in enumerate(self.segtransforms):
            if isinstance(t, strong_img_aug):
                image = t(image)
            else:
                image, label = t(image, label)
        return image, label


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1 <= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            # 走这里
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, scales)
        return img

def get_augment_list(flag_using_wide=False):
    if flag_using_wide:
        l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.1, 1.8]),
        (img_aug_brightness, [0.1, 1.8]),
        (img_aug_color, [0.1, 1.8]),
        (img_aug_sharpness, [0.1, 1.8]),
        (img_aug_posterize, [2, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
        ]
    else:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5])
        ]
    return l


import collections
from collections.abc import Iterable


class Resize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        # assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2
        assert isinstance(ratio_range, Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_label):
        w, h = in_image.size

        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]),
                                           int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size

            # obtain new oh, ow
            if self.bigger_side_to_base_size:
                if h > w:
                    oh = long_side
                    ow = int(1.0 * long_side * w / h + 0.5)
                else:
                    oh = int(1.0 * long_side * h / w + 0.5)
                    ow = long_side
            else:
                oh, ow = (long_side, int(1.0 * long_side * w / h + 0.5)) if h < w else (
                    int(1.0 * long_side * h / w + 0.5), long_side)

            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            return image, label
        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                # print("="*100, h, self.base_size[0])
                # print("="*100, w, self.base_size[1])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size
            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            # print("="*100, in_image.size, image.size)
            return image, label

        else:
            raise ValueError


class RandomFlip(object):
    def __init__(self, prob=0.5, flag_hflip=True, ):
        self.prob = prob
        if flag_hflip:
            self.type_flip = Image.FLIP_LEFT_RIGHT
        else:
            self.type_flip = Image.FLIP_TOP_BOTTOM

    def __call__(self, in_image, in_label):
        if random.random() < self.prob:
            in_image = in_image.transpose(self.type_flip)
            in_label = in_label.transpose(self.type_flip)
        return in_image, in_label

import cv2
class Crop(object):
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError

        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label):
        # Padding to return the correct crop size
        w, h = in_image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32),
                                       value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(np.asarray(in_label, dtype=np.int32),
                                       value=self.ignore_value, **pad_kwargs)
            image = Image.fromarray(np.uint8(image))
            label = Image.fromarray(np.uint8(label))
        else:
            image = in_image
            label = in_label

        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2
        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        label = label.crop((x, y, x + self.crop_w, y + self.crop_h))
        return image, label

def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # # print(f"final:{v}")
    # v = np.random.uniform(scale[0], scale[1])
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)