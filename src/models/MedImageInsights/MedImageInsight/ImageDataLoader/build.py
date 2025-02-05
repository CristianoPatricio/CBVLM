from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import pathlib
from os.path import basename

from timm.data import create_loader
import torch
import torch.utils.data
import torch.distributed as dist
import torchvision.datasets as datasets
from torchvision.io import read_image
import torch.distributed as dist
from pathlib import Path
from yacs.config import CfgNode as CN

from ..LangEncoder import build_tokenizer

from .tsv import TSVImageTextDatasetV2
from .tsv import TSVMeta
from .transforms import build_transforms

logger = logging.getLogger(__name__)


def build_dataset(cfg, is_train):
    if cfg['DATASET']['DATASET'] == 'image_text_pairs_v2':
        dataset = _build_pairs_dataset_v2(cfg, is_train)
    else:
        raise ValueError(f'Unknown dataset: {cfg["DATASET"]["DATASET"]}')
    return dataset


def _get_tsv_list(cfg, is_train):
    tmp_list = []
    if is_train and 'TRAIN_TSV_LIST' in cfg['DATASET']:
        tmp_list = cfg['DATASET']['TRAIN_TSV_LIST']
    elif 'TEST_TSV_LIST' in cfg['DATASET']:
        tmp_list = cfg['DATASET']['TEST_TSV_LIST']

    tsv_list = []
    for l in tmp_list:
        if l.endswith('.list'):
            with open(l, 'r') as f:
                tsv_list.extend([i.strip() for i in f])
        else:
            tsv_list.append(l)

    logger.info(f'tsv list: {tsv_list}')

    return tsv_list


def _get_token_file(cfg):
    num_nodes = dist.get_world_size() // torch.cuda.device_count()
    if isinstance(cfg['DATASET']['TOKEN_FILE'], list):
        if num_nodes == 1:
            logger.warning('=> Multi token files are provided, but only one node is used for training')
            sas_token_file = cfg['DATASET']['TOKEN_FILE'][0]
        else:
            rank = dist.get_rank()
            node_idx = rank // torch.cuda.device_count()
            num_token_files = len(cfg['DATASET']['TOKEN_FILE'])
            sas_token_file = cfg['DATASET']['TOKEN_FILE'][node_idx % num_token_files]
    else:
        sas_token_file = cfg['DATASET']['TOKEN_FILE']

    sas_token_file = os.path.join(cfg['DATASET']['ROOT'], sas_token_file)

    if (
            cfg['DATASET']['LOADER'] == 'blobfuse'
            or not os.path.isfile(sas_token_file)
    ):
        sas_token_file = None

    return sas_token_file


def _build_pairs_dataset_v2(cfg, is_train):
    transforms = build_transforms(cfg, is_train)
    logger.info('transforms: {}'.format(transforms))

    dataset_name = cfg['DATASET']['TRAIN_SET'] \
        if is_train else cfg['DATASET']['TEST_SET']

    tokenobj = build_tokenizer(cfg['LANG_ENCODER'])

    if cfg['DATASET']['DATA_FORMAT'] != 'tsv':
        raise ValueError('Only support tsv format for pairs dataset v2')

    tsv_list = _get_tsv_list(cfg, is_train)

    if len(tsv_list) > 0:
        tsv_filenames = sorted(
            [
                os.path.join(cfg['DATASET']['ROOT'], dataset_name, f)
                for f in tsv_list
            ]
        )
    else:
        dataset_path = os.path.join(cfg['DATASET']['ROOT'], dataset_name)
        tsv_files = Path(dataset_path).glob('**/*.tsv')

        tsv_filenames = sorted(
            [
                str(path)
                for path in tsv_files
            ]
        )

    image_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                'image-' in basename(filename)
                or 'image_' in basename(filename)
                or '_image' in basename(filename)
                or '-image' in basename(filename)
                or 'images-' in basename(filename)
        )
    ]
    text_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                'text-' in basename(filename)
                or 'text_' in basename(filename)
                or '_text' in basename(filename)
                or '-text' in basename(filename)
                or 'texts-' in basename(filename)
        )
    ]

    logger.info(
        "=> found %d/%d tsv file(s) to load.",
        len(image_tsv_files), len(text_tsv_files)
    )

    num_captions = 1 \
        if is_train else cfg['DATASET'].get('NUM_CAPTIONS', 1)
    text_format = cfg['DATASET'].get('TEXT_FORMAT', 'json')

    sas_token_file = _get_token_file(cfg)
    logger.info("=> SAS token path: %s", sas_token_file)

    metas = []
    cfg_data = cfg['DATASET']
    if 'CLASSIFICATION_SETS' in cfg_data and 'NUM_CLASSES' in cfg_data:
        for source, num_classes in zip(cfg_data['CLASSIFICATION_SETS'], cfg_data['NUM_CLASSES']):
            metas.append(
                TSVMeta(
                    source=source,
                    num_classes=num_classes,
                    task='classification'
                )
            )
            logger.info('=> add meta: {}'.format(metas[-1]))

    if 'coco-caption' in dataset_name:
        logger.info('=> coco caption data is used')
        logger.info('=> update num_captions: 5, text_format: json')
        logger.warning('=> set sas token to None for coco evaluation')
        sas_token_file = None
        num_captions = 5
        text_format = 'json'

    dataset = TSVImageTextDatasetV2(
        image_tsv_files, text_tsv_files,
        transform=transforms,
        tokenize=tokenobj,
        context_length=cfg['LANG_ENCODER']['CONTEXT_LENGTH'],
        num_captions=num_captions,
        text_format=text_format,
        is_train=is_train,
        sas_token_path=sas_token_file,
        metas=metas,
        prompt_engineering=cfg['DATASET'].get('PROMPT_ENGINEERING', True),
        concat_queries=cfg['DATASET'].get('CONCAT_QUERIES', False)
    )

    logger.info(
        "=> %s set size: %d", 'train'
        if is_train else 'val', len(dataset)
    )

    return dataset


def build_dataloader(cfg, is_train=True, distributed=False):
    dataset = build_dataset(cfg, is_train)

    if (
            is_train
            and 'TIMM_AUG' in cfg['AUG']
            and cfg['AUG']['TIMM_AUG']['USE_LOADER']
    ):
        logger.info('=> use timm loader for training')
        timm_cfg = CN(init_dict=cfg['AUG']['TIMM_AUG'])
        data_loader = create_loader(
            dataset,
            input_size=cfg['IMAGE_ENCODER']['IMAGE_SIZE'][0],
            batch_size=cfg['TRAIN']['BATCH_SIZE_PER_GPU'],
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            re_split=timm_cfg.RE_SPLIT,
            scale=cfg['AUG']['SCALE'],
            ratio=cfg['AUG']['RATIO'],
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            num_aug_splits=0,
            interpolation=cfg['AUG']['INTERPOLATION'],
            mean=cfg['IMAGE_ENCODER']['IMAGE_MEAN'],
            std=cfg['IMAGE_ENCODER']['IMAGE_STD'],
            num_workers=cfg['WORKERS'],
            distributed=distributed,
            collate_fn=None,
            pin_memory=cfg['PIN_MEMORY'],
            use_multi_epochs_loader=True
        )
    else:
        if is_train:
            batch_size_per_gpu = cfg['TRAIN']['BATCH_SIZE_PER_GPU']
            shuffle = cfg['TRAIN'].get('SHUFFLE', True)
        else:
            batch_size_per_gpu = cfg['TEST']['BATCH_SIZE_PER_GPU']
            shuffle = cfg['TEST'].get('SHUFFLE', False)

        if distributed or cfg.get('ALWAYS_ENABLE_SAMPLER', False):
            # sampler = build_sampler(cfg, dataset, is_train, shuffle)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=cfg['WORKERS'],
            pin_memory=cfg['PIN_MEMORY'],
            sampler=sampler,
            drop_last=True if is_train else False,
            prefetch_factor=cfg.get('PREFETCH_FACTOR', 2)
        )

    return data_loader




