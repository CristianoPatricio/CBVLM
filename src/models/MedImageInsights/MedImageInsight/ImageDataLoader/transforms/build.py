from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timm
from timm.data import create_transform

from yacs.config import CfgNode as CN
from PIL import ImageFilter
import logging
import random

import torch
import torchvision.transforms as T


from .autoaugment import AutoAugmentPolicy
from .autoaugment import AutoAugment
from .autoaugment import RandAugment
from .autoaugment import TrivialAugmentWide
from .threeaugment import deitIII_Solarization
from .threeaugment import deitIII_gray_scale
from .threeaugment import deitIII_GaussianBlur

from PIL import ImageOps
from timm.data.transforms import RandomResizedCropAndInterpolation

logger = logging.getLogger(__name__)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96*96 else (512, 480)


INTERPOLATION_MODES = {
    'bilinear': T.InterpolationMode.BILINEAR,
    'bicubic': T.InterpolationMode.BICUBIC,
    'nearest': T.InterpolationMode.NEAREST,
}


def build_transforms(cfg, is_train=True):
    # assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    normalize = T.Normalize(
        mean=cfg['IMAGE_ENCODER']['IMAGE_MEAN'],
        std=cfg['IMAGE_ENCODER']['IMAGE_STD']
    )

    transforms = None
    if is_train:
        if 'THREE_AUG' in cfg['AUG']:
            img_size = cfg['IMAGE_ENCODER']['IMAGE_SIZE']
            remove_random_resized_crop = cfg['AUG']['THREE_AUG']['SRC']
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            primary_tfl = []
            scale=(0.08, 1.0)
            interpolation='bicubic'
            if remove_random_resized_crop:
                primary_tfl = [
                    T.Resize(img_size, interpolation=3),
                    T.RandomCrop(img_size, padding=4,padding_mode='reflect'),
                    T.RandomHorizontalFlip()
                ]
            else:
                primary_tfl = [
                    RandomResizedCropAndInterpolation(
                        img_size, scale=scale, interpolation=interpolation),
                    T.RandomHorizontalFlip()
                ]
            secondary_tfl = [T.RandomChoice([gray_scale(p=1.0),
                                             Solarization(p=1.0),
                                             GaussianBlurDeiTv3(p=1.0)])]
            color_jitter = cfg['AUG']['THREE_AUG']['COLOR_JITTER']
            if color_jitter is not None and not color_jitter==0:
                secondary_tfl.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))
            final_tfl = [
                    T.ToTensor(),
                    T.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ]
            return T.Compose(primary_tfl+secondary_tfl+final_tfl)
        elif 'TIMM_AUG' in cfg['AUG'] and cfg['AUG']['TIMM_AUG']['USE_TRANSFORM']:
            logger.info('=> use timm transform for training')
            timm_cfg = cfg['AUG']['TIMM_AUG']
            transforms = create_transform(
                input_size=cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0],
                is_training=True,
                use_prefetcher=False,
                no_aug=False,
                re_prob=timm_cfg.get('RE_PROB', 0.),
                re_mode=timm_cfg.get('RE_MODE', 'const'),
                re_count=timm_cfg.get('RE_COUNT', 1),
                re_num_splits= 0 if not timm_cfg.get('RE_SPLITS', False) else timm_cfg['RE_SPLITS'], # if false or 0, return 0
                scale=cfg['AUG'].get('SCALE', None),
                ratio=cfg['AUG'].get('RATIO', None),
                hflip=timm_cfg.get('HFLIP', 0.5),
                vflip=timm_cfg.get('VFLIP', 0.),
                color_jitter=timm_cfg.get('COLOR_JITTER', 0.4),
                auto_augment=timm_cfg.get('AUTO_AUGMENT', None),
                interpolation=cfg['AUG']['INTERPOLATION'],
                mean=cfg['IMAGE_ENCODER']['IMAGE_MEAN'],
                std=cfg['IMAGE_ENCODER']['IMAGE_STD'],
            )
        elif 'TORCHVISION_AUG' in cfg['AUG']:
            logger.info('=> use torchvision transform fro training')
            crop_size = cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0]
            interpolation = INTERPOLATION_MODES[cfg['AUG']['INTERPOLATION']]
            trans = [
                T.RandomResizedCrop(
                    crop_size, scale=cfg['AUG']['SCALE'], ratio=cfg['AUG']['RATIO'],
                    interpolation=interpolation
                )
            ]
            hflip_prob = cfg['AUG']['TORCHVISION_AUG']['HFLIP']
            auto_augment_policy = cfg['AUG']['TORCHVISION_AUG'].get('AUTO_AUGMENT', None)
            if hflip_prob > 0:
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            if auto_augment_policy is not None:
                if auto_augment_policy == "ra":
                    trans.append(RandAugment(interpolation=interpolation))
                elif auto_augment_policy == "ta_wide":
                    trans.append(TrivialAugmentWide(interpolation=interpolation))
                else:
                    aa_policy = AutoAugmentPolicy(auto_augment_policy)
                    trans.append(AutoAugment(policy=aa_policy, interpolation=interpolation))
            trans.extend(
                [
                    T.ToTensor(),
                    normalize,
                ]
            )
            random_erase_prob = cfg['AUG']['TORCHVISION_AUG']['RE_PROB']
            random_erase_scale = cfg['AUG']['TORCHVISION_AUG'].get('RE_SCALE', 0.33)
            if random_erase_prob > 0:
                # NCFC (4/26/2023): Added scale parameter to random erasing for medical imaging
                trans.append(T.RandomErasing(p=random_erase_prob, scale = (0.02, random_erase_scale)))

            from torchvision.transforms import InterpolationMode
            rotation = cfg['AUG']['TORCHVISION_AUG'].get('ROTATION', 0.0)
            if (rotation > 0.0):
                trans.append(T.RandomRotation(rotation, interpolation=InterpolationMode.BILINEAR))
                logger.info(" TORCH AUG: Rotation: " + str(rotation))

            transforms = T.Compose(trans)
        elif cfg['AUG'].get('RANDOM_CENTER_CROP', False):
            logger.info('=> use random center crop data augmenation')
            # precrop, crop = get_resolution(cfg.TRAIN.IMAGE_SIZE)
            crop = cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0]
            padding = cfg['AUG'].get('RANDOM_CENTER_CROP_PADDING', 32)
            precrop = crop + padding
            mode = INTERPOLATION_MODES[cfg['AUG']['INTERPOLATION']]
            transforms = T.Compose([
                T.Resize(
                    (precrop, precrop),
                    interpolation=mode
                ),
                T.RandomCrop((crop, crop)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        elif cfg['AUG'].get('MAE_FINETUNE_AUG', False):
            mean = cfg['IMAGE_ENCODER']['IMAGE_MEAN']
            std = cfg['IMAGE_ENCODER']['IMAGE_STD']
            transforms = create_transform(
                input_size=cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0],
                is_training=True,
                color_jitter=cfg['AUG'].get('COLOR_JITTER', None),
                auto_augment=cfg['AUG'].get('AUTO_AUGMENT', 'rand-m9-mstd0.5-inc1'),
                interpolation='bicubic',
                re_prob=cfg['AUG'].get('RE_PROB', 0.25),
                re_mode=cfg['AUG'].get('RE_MODE', "pixel"),
                re_count=cfg['AUG'].get('RE_COUNT', 1),
                mean=mean,
                std=std,
            )
        elif cfg['AUG'].get('MAE_PRETRAIN_AUG', False):
            mean = cfg['IMAGE_ENCODER']['IMAGE_MEAN']
            std = cfg['IMAGE_ENCODER']['IMAGE_STD']
            transforms = T.Compose([
                T.RandomResizedCrop(cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0], scale=tuple(cfg['AUG']['SCALE']), interpolation=INTERPOLATION_MODES["bicubic"]),  # 3 is bicubic
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)])
        elif cfg['AUG'].get('ThreeAugment', False): # from DeiT III
            mean = cfg['IMAGE_ENCODER']['IMAGE_MEAN']
            std = cfg['IMAGE_ENCODER']['IMAGE_STD']
            img_size = cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0]
            remove_random_resized_crop = cfg['AUG'].get('src', False)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            primary_tfl = []
            scale=(0.08, 1.0)
            interpolation='bicubic'
            if remove_random_resized_crop:
                primary_tfl = [
                    T.Resize(img_size, interpolation=3), # bicubic
                    T.RandomCrop(img_size, padding=4,padding_mode='reflect'),
                    T.RandomHorizontalFlip()
                ]
            else:
                primary_tfl = [
                    timm.data.transforms.RandomResizedCropAndInterpolation(
                        img_size, scale=scale, interpolation=interpolation),
                    T.RandomHorizontalFlip()
                ]

            secondary_tfl = [T.RandomChoice([deitIII_gray_scale(p=1.0),
                                             deitIII_Solarization(p=1.0),
                                             deitIII_GaussianBlur(p=1.0)])]
            color_jitter = cfg['AUG']['COLOR_JITTER']
            secondary_tfl.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))
            final_tfl = [
                    T.ToTensor(),
                    T.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ]
            transforms = T.Compose(primary_tfl+secondary_tfl+final_tfl)
        logger.info('=> training transformers: {}'.format(transforms))
    else:
        mode = INTERPOLATION_MODES[cfg['AUG']['INTERPOLATION']]
        if cfg['TEST']['CENTER_CROP']:
            transforms = T.Compose([
                T.Resize(
                    int(cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0] / 0.875),
                    # the same behavior as in deit: size = int((256 / 224) * args.input_size)
                    # 224 / 256 = 0.875
                    interpolation=mode
                ),
                T.CenterCrop(cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0]),
                T.ToTensor(),
                normalize,
            ])
        else:
            transforms = T.Compose([
                T.Resize(
                    (cfg['IMAGE_ENCODER']['IMAGE_SIZE'][1], cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0]),
                    interpolation=mode
                ),
                T.ToTensor(),
                normalize,
            ])
        logger.info('=> testing transformers: {}'.format(transforms))

    return transforms

