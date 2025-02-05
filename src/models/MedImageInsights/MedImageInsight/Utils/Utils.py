import logging
import os
import torch
import torch.distributed as dist
import yaml

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from fvcore.nn import flop_count_str


logger = logging.getLogger(__name__)


NORM_MODULES = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
]

def register_norm_module(cls):
    NORM_MODULES.append(cls)

    return cls


def is_main_process():
    rank = 0
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    return rank == 0


@torch.no_grad()
def analysis_model(model, dump_input, verbose=False):
    model.eval()
    flops = FlopCountAnalysis(model, dump_input)
    total = flops.total()
    model.train()
    params_total = sum(p.numel() for p in model.parameters())
    params_learned = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"flop count table:\n {flop_count_table(flops)}")
    if verbose:
        logger.info(f"flop count str:\n {flop_count_str(flops)}")
    logger.info(f"  Total flops: {total/1000/1000:.3f}M,")
    logger.info(f"  Total params: {params_total/1000/1000:.3f}M,")
    logger.info(f"  Learned params: {params_learned/1000/1000:.3f}M")

    return total, flop_count_table(flops), flop_count_str(flops)


def load_config_dict_to_opt(opt, config_dict, splitter='.'):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split(splitter)
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            print(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_file(conf_file):
    """
    Load opt from the config file.

    Args:
        conf_file: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        load_config_dict_to_opt(opt, config_dict)

    return opt

def cast_batch_to_dtype(batch, dtype):
    """
    Cast the float32 tensors in a batch to a specified torch dtype.
    It should be called before feeding the batch to the FP16 DeepSpeed model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
    Returns:
        return_batch: same type as the input batch with internal float32 tensors casted to the specified dtype.
    """
    if torch.is_tensor(batch):
        if torch.is_floating_point(batch):
            return_batch = batch.to(dtype)
        else:
            return_batch = batch
    elif isinstance(batch, list):
        return_batch = [cast_batch_to_dtype(t, dtype) for t in batch]
    elif isinstance(batch, tuple):
        return_batch = tuple(cast_batch_to_dtype(t, dtype) for t in batch)
    elif isinstance(batch, dict):
        return_batch = {}
        for k in batch:
            return_batch[k] = cast_batch_to_dtype(batch[k], dtype)
    else:
        logger.debug(f"Can not cast type {type(batch)} to {dtype}. Skipping it in the batch.")
        return_batch = batch

    return return_batch


def cast_batch_to_half(batch):
    """
    Cast the float32 tensors in a batch to float16.
    It should be called before feeding the batch to the FP16 DeepSpeed model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
    Returns:
        return_batch: same type as the input batch with internal float32 tensors casted to float16
    """
    return cast_batch_to_dtype(batch, torch.float16)
