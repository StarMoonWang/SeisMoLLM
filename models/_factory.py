import sys
import os
import warnings
import torch
from typing import Callable


__all__ = [
    "get_model_list",
    "register_model",
    "create_model",
    "save_checkpoint",
    "load_checkpoint",
]


_name_to_creators = {}


def get_model_list():
    return list(_name_to_creators)


def register_model(func: Callable) -> Callable:
    """Register a model."""
    model_name = func.__name__

    if model_name in _name_to_creators:
        raise Exception(f"Model '{model_name}' already exists.")

    model_module = sys.modules[func.__module__]

    if hasattr(model_module, "__all__") and model_name not in model_module.__all__:
        model_module.__all__.append(model_name)

    _name_to_creators[model_name] = func

    return func


def create_model(model_name: str, **kwargs):
    """Create a model.

    Args:
        model_name (str): Model name.
    """

    if model_name not in _name_to_creators:
        avl_model_names = get_model_list()
        raise ValueError(f"Model '{model_name}' does not exist. \nAvailable: {avl_model_names}")

    Model = _name_to_creators[model_name]

    model = Model(**kwargs)

    return model


def save_checkpoint(
    save_path: str, epoch: int, model, optimizer, scheduler, best_loss: float
) -> None:
    """Save checkpoint."""
    if hasattr(model, "module"):
        model_without_ddp = model.module
        use_ddp = True
    else:
        model_without_ddp = model
        use_ddp = False

    if hasattr(model, "_orig_mod"):
        model_without_compile = model_without_ddp._orig_mod
        use_compile = True
    else:
        model_without_compile = model_without_ddp
        use_compile = False
        
    torch.save(
        {
            "epoch": epoch,
            "model_dict": model_without_compile.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            "loss": best_loss,
            "use_compile": use_compile,
            "use_ddp": use_ddp,
        },
        save_path,
    )


def load_checkpoint(
    save_path: str,
    device: torch.device,
    dist_mode=False,
    compile_mode=False,
    resume=False,
):
    """Load checkpoint."""
    checkpoint = torch.load(save_path, map_location=device)
    
    
    if "model_dict" not in checkpoint:
        checkpoint = {"model_dict":checkpoint}

    checkpoint["model_dict"] = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in checkpoint["model_dict"].items()
    }

    if resume:
        used_ddp = checkpoint.get("use_ddp", False)
        used_compile = checkpoint.get("use_compile", False)
        if used_ddp != dist_mode:
            msg = (
                f"The model was trained {'without' if not used_ddp else 'with'} using distributed mode, "
                f"but distributed mode is {'enabled' if dist_mode else 'disabled'} now, "
                f"which may lead to unreproducible results."
            )
            warnings.warn(msg)
        if used_compile != compile_mode:
            msg = (
                f"The model was trained {'without' if not used_compile else 'with'} using `torch.compile` (triton random), "
                f"but the argument `use_torch_compile` is `{compile_mode}`, which may lead to unreproducible results."
            )
            warnings.warn(msg)

    return checkpoint
