import math
import logging
import copy
import itertools
import random
from collections.abc import Iterable, Iterator
import torch
from torch._C import default_generator
import torch.distributed as dist
import time
from functools import wraps, partial

logger = logging.getLogger(__name__)


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n / decay)  # exponential decay over 100 updates
            self.sum = alpha * self.sum + (1 - alpha) * val * n
            self.count = alpha * self.count + (1 - alpha) * n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def getstate(self):
        return {'val': self.val,
                'avg': self.avg,
                'sum': self.sum,
                'count': self.count}

    def setstate(self, state):
        self.val = state['val']
        self.avg = state['avg']
        self.sum = state['sum']
        self.count = state['count']


def move_batch_to_device(batch, device):
    """
    Move the batch to the device.
    It should be called before feeding the batch to the model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
        device (torch.device): device to move the batch to
    Returns:
        return_batch: same type as the input batch with internal tensors moved to device
    """
    if torch.is_tensor(batch):
        return_batch = batch.to(device)
    elif isinstance(batch, list):
        return_batch = [move_batch_to_device(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return_batch = tuple(move_batch_to_device(t, device) for t in batch)
    elif isinstance(batch, dict):
        return_batch = {}
        for k in batch:
            return_batch[k] = move_batch_to_device(batch[k], device)
    else:
        logger.debug(f"Can not move type {type(batch)} to device. Skipping it in the batch.")
        return_batch = batch

    return return_batch


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


def cast_batch_to_bf16(batch):
    """
    Cast the float32 tensors in a batch to bfloat16.
    It should be called before feeding the batch to the FP16 DeepSpeed model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
    Returns:
        return_batch: same type as the input batch with internal float32 tensors casted to bfloat16
    """
    return cast_batch_to_dtype(batch, torch.bfloat16)


# copied from MainzSpeech/moe_tools
def peek_first_item_from_iterator(it):
    # extract first item from iterator
    first_item = next(it)
    # create iterator with the first item added back in
    new_it = itertools.chain([copy.deepcopy(first_item)], it)
    return first_item, new_it


# copied from MainzSpeech/moe_tools
def generate_dummy_batch(it):
    """
    Generates a dummy batch by peeking at given iterable or iterator on rank 0,
    then broadcast dummy_batch to all other ranks.
    """
    from mpi4py import MPI
    assert isinstance(it, Iterable) or isinstance(it, Iterator)
    if isinstance(it, Iterable):
        it = iter(it)
    if MPI.COMM_WORLD.Get_rank() == 0:
        dummy_batch, it = peek_first_item_from_iterator(it)
    else:
        dummy_batch = None
    dummy_batch = MPI.COMM_WORLD.bcast(dummy_batch, root=0)
    assert dummy_batch is not None
    return dummy_batch, it


def retry_on_failure(func=None, *, max_retries=3, on_error_func=None, on_retry_func=None, raise_err_func=None, sleep_time=30, error_types=(Exception,)):
    """
    Decorator utility to retry a function, this decorator must be used without arguments (@retry_on_failure) or with all named arguments (@retry_on_failure(max_retries=10)).
    Args:
        max_retries (int): The number of retries to perform, in addition to the initial retry. Defaults to 3.
        sleep_time (int): The time in seconds to wait before the next retry. Defaults to 30.
        error_types (Tuple[type]): a tuple of exception types which are used to except any error being retried, if the exception that is thrown is not an instance of one of these types, the function is not retried. Defaults to (Exception,) which covers all exceptions.
        on_retry_func (callable(num_retries)): A function with a single argument, the number of retries done so far. This function is called just before any retry. Defaults to a function logging `num_retries`.
        on_error_func (callable(num_retries)): A function with a single argument, the number of retries done in total. This function is called after `max_retries` has been tried. Defaults to a function logging `num_retries`.
        raise_err_func (callable(err)): A function with a single argument, the exception that was thrown. This function is called after `max_retries` has been tried. Defaults to raising the error.
    """
    if on_error_func is None:
        def on_error_func(retried_times):
            logger.warning(f"Failed after retrying {retried_times} times")

    if on_retry_func is None:
        def on_retry_func(idx):
            logger.warning(f"Retrying on failure {idx}")

    if raise_err_func is None:
        def raise_err_func(err):
            raise err

    if func is None:
        return partial(
            retry_on_failure,
            max_retries=max_retries,
            on_error_func=on_error_func,
            on_retry_func=on_retry_func,
            raise_err_func=raise_err_func,
            sleep_time=sleep_time,
            error_types=error_types,
        )

    @wraps(func)
    def decorator(*args, **kwargs):
        num_retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except error_types as err:
                num_retries += 1
                on_retry_func(num_retries)
                if num_retries > max_retries:
                    on_error_func(num_retries)
                    raise_err_func(err)
                time.sleep(sleep_time)

    return decorator


class TemporaryRngState:
    '''
    Context manager for working with a temporary random number generator (RNG) state.
    The constructor gets a random number from the Python RNG that is used as
    (part of) the seed for the temporary RNG
    and then stores the current RNG state to restore the it later on.
    If add_rank_to_seed=True, the GPU rank is added to the seed.
    This is useful to initialize MoE models
    where the experts on different GPUs should be initialized independently.
    Note that this feature requires torch.distributed to be initialized.
    On enter, the context managers sets the RNG state to the random seed created in the constructor
    to establish a temporary RNG state.
    On exit, the context manager resets the RNG state to the previously remembered state.
    Thereby, any RNG operations executed with this context manager
    do not affect the global, non-temporary RNG state.
    However, the usage of this context manager does advance the Python RNG
    since it uses that RNG to generate the random seed in the constructor.
    The context manager resets the Python RNG state and
    the PyTorch RNG state for CPU and GPU (if cuda is initialized).
    It does not currently reset the numpy RNG state.
    '''
    def __init__(self, add_rank_to_seed=False):
        self.seed = random.randrange(2**32)
        if add_rank_to_seed and dist.is_initialized():
            self.seed += dist.get_rank()
        self.python_rng_state = random.getstate()
        self.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_initialized():
            self.torch_rng_state_cuda = torch.cuda.get_rng_state()

    def __enter__(self):
        # increment seed for different RNGs to avoid correlation
        # in the (very unlikely) case that the different RNGs
        # use the exact same algorithm
        random.seed(self.seed)
        # do not call torch.maunal_seed here, because that sets the seed of all GPUs
        default_generator.manual_seed(self.seed + 1)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed(self.seed + 2)  # only set seed of default cuda device

    def __exit__(self, exc_type, exc_value, exc_traceback):
        random.setstate(self.python_rng_state)
        torch.set_rng_state(self.torch_rng_state)
        if torch.cuda.is_initialized():
            torch.cuda.set_rng_state(self.torch_rng_state_cuda)
