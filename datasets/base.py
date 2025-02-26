import pandas as pd
from typing import Optional,Tuple
import copy

class DatasetBase:
    """
    The base class for datasets.

    """

    _name: str = ""
    _part_range: Optional[tuple] = None
    _channels: list = []
    _sampling_rate: int = 0

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
    ):
        """
        Args:
            seed: int
                Random seed.
            mode: str
                train / val /test
            data_dir: str
                Directory of dataset.
            shuffle: bool
                If true, meta data will be shuffled.
            data_split: bool
                whether split dataset to train/val/test
            train_size: float
                size of train set
            val_size: float
                size of validation set
        """
        self._seed = seed

        assert mode.lower() in ["train", "val", "test"]
        self._mode = mode.lower()

        self._data_dir = data_dir
        self._shuffle = shuffle
        self._data_split = data_split

        assert (
            train_size + val_size < 1.0
        ), f"train_size:{train_size}, val_size:{val_size}"
        self._train_size = train_size
        self._val_size = val_size

        self._meta_data = self._load_meta_data()

    

    def _load_meta_data(self, filename = None) -> pd.DataFrame:
        pass

    def _load_event_data(self, idx: int) -> dict:
        pass 
    
    def __repr__(self) -> str:
        return (
            f"Dataset(name:{self._name}, part_range:{self._part_range}, channels:{self._channels}, "
            f"sampling_rate:{self._sampling_rate}, data_dir:{self._data_dir}, shuffle:{self._shuffle}, "
            f"data_split:{self._data_split}, train_size:{self._train_size}, val_size:{self._val_size})"
        )

    def __len__(self):
        return len(self._meta_data)
    
    def __getitem__(self,idx:int)->Tuple[dict,dict]:
        return self._load_event_data(idx=idx)

    @classmethod
    def name(cls):
        return cls._name

    @classmethod
    def sampling_rate(cls):
        return cls._sampling_rate

    @classmethod
    def channels(cls):
        return copy.deepcopy(cls._channels)
