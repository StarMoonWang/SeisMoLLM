from .base import DatasetBase
from typing import Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from utils import logger
from ._factory import register_dataset

__all__ = ["STEAD", "STEAD_mag"]


class STEAD(DatasetBase):
    '''This may not work for downloaded original STEAD, 
    as I spilt it into multiple slices for faster loading'''
    _part_range = None
    _name = "stead"
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self,filename = "eq.csv") -> pd.DataFrame:
        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
                dtype={
                        "file_idx": int,
                        "trace_name": str,
                        "trace_category": str,
                        "source_id": str,
                        "source_magnitude": np.float32,
                        "source_magnitude_type": str,
                        "p_arrival_sample": np.float32,
                        "p_travel_sec": np.float32,
                        "s_arrival_sample": np.float32,
                        "reciever_code": str,
                        "source_distance_km": np.float32,
                        "source_distance_deg": np.float32,
                        "back_azimuth_deg": np.float32,
                        "snr_db": str
                    },
                    low_memory=False
                )

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0] : r[1], :]
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df
    

    def _load_event_data(self, idx:int) -> Tuple[dict,dict]:
        """Load evnet data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """        
    
        target_event = self._meta_data.iloc[idx]
        file, trace = target_event['file_idx'], target_event['trace_name']

        path = os.path.join(self._data_dir, f"part_{file}.hdf5")
        with h5py.File(path, "r") as f:
            dataset = f.get("data/" + str(trace))
            data = np.array(dataset).astype(np.float32).T

        (   ppk,spk,
            mag_type,evmag,
            baz, dis, ptt, snr
        ) = itemgetter(
            "p_arrival_sample","s_arrival_sample",
            "source_magnitude_type","source_magnitude",
            "back_azimuth_deg", "source_distance_km",
            "p_travel_sec", "snr_db" 
            )(
            target_event
            )
        
        ppk, spk = int(ppk), int(spk)

        if pd.notnull(baz):
            baz = baz%360

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)

        def convert_snr(snr):
            if snr == 'nan':
                return [0.0, 0.0, 0.0]
            else:
                numbers = snr.strip('[]').split()
                return [float(num) for num in numbers]
        snr = np.array(convert_snr(snr))
        
        event = {
            "data": data,
            "ppks": [ppk] if pd.notnull(ppk) else [],
            "spks": [spk] if pd.notnull(spk) else [],
            "emg" : [evmag] if pd.notnull(evmag) else [],
            "baz" : [baz] if pd.notnull(baz) else [],
            "dis" : [dis] if pd.notnull(dis) else [],
            "ptt" : [ptt] if pd.notnull(ptt) else [],
            "snr" : snr,
        }
        target = target_event.to_dict()

        return event,target


class STEAD_mag(STEAD):
    # Only includes earthquake samples with ml magnitude type
    _name = "stead_mag"
    _part_range = None
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self,filename = "eq_MLmag.csv") -> pd.DataFrame:

        return super()._load_meta_data(filename=filename)

    def _load_event_data(self, idx: int) -> Tuple[dict,dict]:
        """Load event data

        Args:
            idx (int): Index of target row.

        Returns:
            dict: Data of event.
            dict: Meta data.
        """        
        return super()._load_event_data(idx=idx)


@register_dataset
def stead(**kwargs):
    dataset = STEAD(**kwargs)
    return dataset


@register_dataset
def stead_mag(**kwargs):
    dataset = STEAD_mag(**kwargs)
    return dataset




'''
source_magnitude_type
ml       724824
md       239233
mb        43174
mw         7284
mb_lg      5420
mpv        4668
mwr        1804
mh         1482
mn          519
mpva        312
h           266
ms          255
mlg         231
mlv         220
m           122
mww         105
mg           84
mdl          76
mblg         63
mbr          27
mlr          22
none         17
mc           13
mwc           7
mwb           3
'''
