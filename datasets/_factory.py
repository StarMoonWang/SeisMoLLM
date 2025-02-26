import sys
from typing import Callable


__all__ = [
    "get_dataset_list",
    "register_dataset",
    "build_dataset"
]


_name_to_creators = {}


def get_dataset_list():
    return list(_name_to_creators)


def register_dataset(func: Callable) -> Callable:
    """Register a dataset."""
    dataset_name = func.__name__

    if dataset_name in _name_to_creators:
        raise Exception(f"Dataset '{dataset_name}' already exists.")

    dataset_module = sys.modules[func.__module__]

    if hasattr(dataset_module, "__all__") and dataset_name not in dataset_module.__all__:
        dataset_module.__all__.append(dataset_name)

    _name_to_creators[dataset_name] = func

    return func


def build_dataset(dataset_name: str, **kwargs):
    """Build a dataset.

    Args:
        dataset_name (str): Dataset name.
    """

    if dataset_name not in _name_to_creators:
        raise ValueError(f"Dataset '{dataset_name}' does not exist.")

    Dataset = _name_to_creators[dataset_name]

    dataset = Dataset(**kwargs)

    return dataset

