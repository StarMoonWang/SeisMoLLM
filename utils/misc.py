import datetime
import warnings
import os
import random
import re
from typing import Any, Dict, List, Literal, Union
import GPUtil
import numpy as np
import math
import torch
import torch.distributed as dist


def setup_seed(seed: int) -> None:
    """Setup seed for torch, numpy and random"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_str() -> str:
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return dtstr


def strftimedelta(td: datetime.timedelta) -> str:
    """Convert `timedelta` to `str`.
    Representation: `'{hours}h {minutes}min {seconds}s'`
    """
    _seconds = int(td.seconds + td.microseconds // 1e6)
    hours = int(td.days * 24 + _seconds // 3600)
    minutes = int(_seconds % 3600 / 60)
    seconds = _seconds % 60
    deltastr = f"{hours}h {minutes}min {seconds}s"
    return deltastr


def get_safe_path(path: str, tag: str = "new") -> str:
    """Get a path that does not exist"""
    if is_main_process():
        d = os.path.split(path)[0]
        if not os.path.exists(d):
            os.makedirs(d)
    if os.path.exists(path):
        _tag = "_" + str(tag).replace(" ", "_")
        path = _tag.join(os.path.splitext(path))
        return get_safe_path(path, tag)
    else:
        return path


def _setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process

    Reference: https://github.com/facebookresearch/detr/blob/main/util/misc.py
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_tensor(
    t: torch.Tensor, op: str = "SUM", barrier: bool = False
) -> torch.Tensor:
    """
    All reduce.
    """
    assert op in ["SUM", "AVG", "PRODUCT", "MIN", "MAX", "PREMUL_SUM"]
    _t = t.clone().detach()
    _op = getattr(dist.ReduceOp, op)
    dist.all_reduce(_t, op=_op)
    if barrier:
        dist.barrier()
    return _t


def gather_tensors_to_list(
    t: torch.Tensor, barrier: bool = False
) -> List[torch.Tensor]:
    """
    Gather tensors to a list.
    """
    _t = t.clone().detach()
    _ts = [torch.zeros_like(_t) for _ in range(get_world_size())]
    dist.all_gather(_ts, _t)

    if barrier:
        dist.barrier()

    return _ts


def broadcast_object(obj: Any, src: int = 0, device: torch.device = None) -> Any:
    """
    Broadcast object from src.
    """
    _obj = [obj]
    dist.broadcast_object_list(_obj, src=src, device=device)
    return _obj.pop()


def init_distributed_mode() -> bool:
    """
    Initialize distributed training (backend: NCCL).
    """

    # Arguments from environment
    required_args = set(["WORLD_SIZE", "RANK", "LOCAL_RANK"])
    if not required_args.issubset(os.environ):
        return False

    # For RTX 40xx Series
    for gpu in GPUtil.getGPUs():
        if re.findall(r"GEFORCE RTX", gpu.name, re.I):
            version = re.findall(r"\d+", gpu.name)
            if version and int(version[0]) > 4000:
                os.environ["NCCL_P2P_DISABLE"] = "1"
                os.environ["NCCL_IB_DISABLE"] = "1"
                os.environ["NCCL_DEBUG"] = "info"
                warnings.warn(
                    f"GPU ({gpu.name}) detected. 'NCCL_P2P' and 'NCCL_IB' are disabled."
                )
                break

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200000))
    dist.barrier()

    _setup_for_distributed(is_main_process())

    return True


# def adjust_learning_rate(args, optimizer, train_steps) -> float:
#     """Adjust learning rate.
#     Args:
#         optimizer: optimizer whose learning rate must be shrunk.
#         train_steps: steps now.
#         decay_op: 'sin' or 'e'
#         shrink_factor: factor in interval (0, 1) to multiply learning rate with. (only used when `decay_op` is 'e')
#     Returns:
#         float: new learning rate.
#     """
#     print("Now lr: ", optimizer.param_groups[0]["lr"])
#     if train_steps < args.warmup_steps:
#         percent = (train_steps +1 ) / args.warmup_steps
#         learning_rate = args.lr * percent
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = learning_rate
#         print("New learning rate:", learning_rate)
#     else:
#         if (train_steps - args.warmup_steps + 1) % args.decay_freq == 0:
#             if args.decay_op == "e":
#                 learning_rate = optimizer.param_groups[0]["lr"] ** args.shrink_factor
#             elif args.decay_op == "sin":
#                 learning_rate = np.sin(optimizer.param_groups[0]["lr"])
#             else:
#                 raise ValueError(f"`decay_op` must be 'e' or 'sin', got '{args.decay_op}'")
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = learning_rate
#             print("New learning rate:", learning_rate)
#     return optimizer.param_groups[0]["lr"]


def strfargs(args, configs) -> str:
    """Convert arguments and configs to string."""

    string = ""
    string += "\nArguments:\n"
    for k, v in args.__dict__.items():
        string += f"{k}: {v}\n"
    string += "\nConfigs:\n"
    for k, v in configs.__dict__.items():
        if not (
            (k.startswith("__") and k.endswith("__"))
            or callable(v)
            or isinstance(v, (classmethod, staticmethod))
        ):
            string += f"{k}: {v}\n"
    return string


def count_parameters(module: torch.nn.Module) -> int:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100
    return total_params, trainable_params, trainable_ratio

def cal_snr(
    data: np.ndarray, pat: int, window: int = 500, method: str = "power"
) -> float:
    """Estimates SNR.

    Args:
        data (np.ndarray): 3 component data. Shape: (C, L)
        pat (int): Phase arrival time.
        window (int, optional): The length of the window for calculating the SNR (in the sample). Defaults to 500.
        method (str): Method to calculate SNR. One of {"power", "std"}. Defaults to "power"

    Returns:
        float: Estimated SNR in db.

    Modified from:
        https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/predictor.py

    """
    pat = int(pat)

    assert window < data.shape[-1] / 2, f"window = {window}, data.shape = {data.shape}"
    assert 0 < pat < data.shape[-1], f"pat = {pat}"


    if (pat + window) <= data.shape[-1]:
        if pat >= window:
            nw = data[:, pat - window : pat]
            sw = data[:, pat : pat + window]
        else:
            window = pat
            nw = data[:, pat - window : pat]
            sw = data[:, pat : pat + window]
    else:
        window = data.shape[-1] - pat
        nw = data[:, pat - window : pat]
        sw = data[:, pat : pat + window]
    
    if method == "power":
        snr = np.mean(sw**2) / (np.mean(nw**2) + 1e-6)
    elif method == "std":
        snr = np.std(sw) / (np.std(nw) + 1e-6)
    else:
        raise Exception(f"Unknown method: {method}")

    snr_db = round(10 * np.log10(snr), 2)

    return snr_db
