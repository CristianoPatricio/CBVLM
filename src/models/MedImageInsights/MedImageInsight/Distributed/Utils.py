import logging
import os
import pickle
import requests
import tenacity
import time
import shutil

import torch
import torch.distributed as dist

from PIL import Image
from torchvision.utils import make_grid


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
    logger.info(f"  Total flops: {total / 1000 / 1000:.3f}M,")
    logger.info(f"  Total params: {params_total / 1000 / 1000:.3f}M,")
    logger.info(f"  Learned params: {params_learned / 1000 / 1000:.3f}M")

    return total, flop_count_table(flops), flop_count_str(flops)


def gather_tensors(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(int(os.environ['WORLD_SIZE']))
    ]

    dist.all_gather(tensors_gather, tensor, async_op=False)
    # need to do this to restore propagation of the gradients
    tensors_gather[int(os.environ['RANK'])] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output


def is_valid_url(url):
    try:
        from urllib import parse
        return parse.urlparse(str(url)).scheme != ''
    except Exception:
        return False


@tenacity.retry(stop=tenacity.stop_after_attempt(3))
def download_file(url, filepath):
    logger.info(f'Downloading from {url} to {filepath.absolute()}.')
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        if r.status_code > 200:
            raise RuntimeError(f'Failed in downloading from {url}, status code {r.status_code}.')

        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f, length=4194304)


class DistributionGridFactory:
    """
    DistributionGrid Factory for helping create, cache and share the DistributionGrid based on the usage.
    The DistributionGrid con be shared cross modules only the when this 3 conditions:
        1. expert parallel group size
        2. expert parallel replica group size,
    are the same.
    """
    distribution_grid_cache = {}

    @classmethod
    def get_distribution_grid(cls,
                              expert_parallel_group_size,
                              expert_parallel_replica_group_size,
                              ddp_type):
        """
        Get the DistributionGrid by the conditions.
        Args:
            expert_parallel_group_size: expert parallel group size
            expert_parallel_replica_group_size: expert parallel replica group size
            ddp_type: distributed data parallel type. "DDP" of the recipe, only allow ddp_type is "MAINZ", "OSS" or "ShardedDDP".

        Returns: new created DistributionGrid or shared DistributionGrid.

        Notes: Currently get_distribution_grid only support "DDP" is "MAINZ", "OSS" or "ShardedDDP".
        """
        # TODO:  Support cases that "DDP" is "FSDP".
        # For "FSDP", we use the DG of self.opt['fsdp_expert_grid'] which is initialize in DistributedTrainer directly.
        ddp_type = ddp_type.upper()
        assert ddp_type in ["MAINZ", "OSS", "SHARDEDDDP"], f'DistributionGrid Factory only support "DDP" is "MAINZ",' \
                                             f' "OSS" or "ShardedDDP".' \
                                             f' But currently "DDP" is {ddp_type}'

        cached_distributed_grid = cls.distribution_grid_cache.get(
            (expert_parallel_group_size, expert_parallel_replica_group_size), None)

        if cached_distributed_grid is not None:
            return cached_distributed_grid
        else:
            from ort_moe.grids import DistributionGrid
            distributed_grid = DistributionGrid(expert_parallel_group_size=expert_parallel_group_size,
                                                expert_parallel_replica_group_size=expert_parallel_replica_group_size)

            cls.distribution_grid_cache[expert_parallel_group_size,
                                        expert_parallel_replica_group_size] = distributed_grid
            return distributed_grid


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        dist.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def all_gather_cpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """

    def _get_global_gloo_group():
        """
        Return a process group based on gloo backend, containing all the ranks
        The result is cached.
        """
        if dist.get_backend() == "nccl":
            return dist.new_group(backend="gloo")
        else:
            return dist.group.WORLD

    if get_world_size() == 1:
        return [data]
    group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def broadcast_data(data):
    if not torch.distributed.is_initialized():
        return data
    rank = dist.get_rank()
    if rank == 0:
        data_tensor = torch.tensor(data + [0], device="cuda")
    else:
        data_tensor = torch.tensor(data + [1], device="cuda")
    torch.distributed.broadcast(data_tensor, 0)
    while data_tensor.cpu().numpy()[-1] == 1:
        time.sleep(1)

    return data_tensor.cpu().numpy().tolist()[:-1]


def reduce_sum(tensor):
    if get_world_size() <= 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def save_result(result, filename):
    output_folder = os.path.dirname(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(output_folder, exist_ok=True)

    if isinstance(result, torch.Tensor) and result.ndim in [3,4]:
        if result.ndim==3 and result.size(0) not in [1,3]:
            result = make_grid(result.unsqueeze(1))
        elif result.ndim==4:
            result = make_grid(result)
        else:
            result = make_grid([result])

        im = Image.fromarray(result.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy())
        im.save(os.path.join(output_folder, '{}.png'.format(basename)))
    else:
        torch.save(result, os.path.join(output_folder, '{}.pth'.format(basename)))
