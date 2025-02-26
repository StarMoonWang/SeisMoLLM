import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from obspy.signal.trigger import trigger_onset
from utils import logger
from config import Config

__all__ = ["process_outputs", "pick_batch", "ResultSaver"]


def _detect_peaks(
    x: np.ndarray,
    mph: int = None,
    mpd: int = 1,
    threshold: float = 0,
    edge: str = "rising",
    kpsh: bool = False,
    valley: bool = False,
    topk: int = None,
) -> np.ndarray:
    """Detect peaks in data based on their amplitude and other features.

    Args:
        x (np.ndarray): data.
        mph (int, optional): detect peaks that are greater than minimum peak height
            (if parameter `valley` is False) or peaks that are smaller than maximum
            peak height (if parameter `valley` is True). Defaults to None.
        mpd (int, optional): detect peaks that are at least separated by minimum peak
            distance (in number of data). Defaults to 1.
        threshold (float, optional): detect peaks (valleys) that are greater (smaller)
            than `threshold` in relation to their immediate neighbors. Defaults to 0.
        edge (str, optional): {None, 'rising', 'falling', 'both'} for a flat peak, keep
            only the rising edge ('rising'), only the falling edge ('falling'), both
            edges ('both'), or don't detect a flat peak (None). Defaults to "rising".
        kpsh (bool, optional): keep peaks with same height even if they are closer
            than `mpd`. Defaults to False.
        valley (bool, optional): if True (1), detect valleys (local minima)
            instead of peaks. Defaults to False.
        topk (int, optional): only the top-k height peaks can be retained. Defaults to None.

    Returns:
        np.ndarray: indeces of the peaks in `x`. like [[11][35] ... [89]]

    Modified from:
        https://github.com/demotu/BMC/blob/master/functions/_detect_peaks.py
    """
    x = np.atleast_1d(x).astype("float32")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        # Top-k inds
        if topk is not None:
            ind = ind[:topk]
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _detect_event(
    outputs: torch.Tensor, prob_threshold: float, topk: int
) -> torch.Tensor:
    """Detect events of one batch.

    Args:
        outputs (torch.Tensor): 2d-Tensor.
        prob_threshold (float): Minimum probability.
        topk (int): Only detect the events whose probabilities are in the top k.

    Returns:
        torch.Tensor: shape (N,in_samples)
    """
    detections = []

    for output in outputs.detach().cpu().numpy():
        detection_indice_pairs = trigger_onset(output, prob_threshold, prob_threshold)
        
        if isinstance(detection_indice_pairs,np.ndarray):
            detection_indice_pairs = detection_indice_pairs.tolist()
        
        detection_indice_pairs.sort(key=lambda v:v[1]-v[0],reverse=True)
        detection_indice_pairs = detection_indice_pairs[:topk]
            
        if len(detection_indice_pairs) < topk:
            detection_indice_pairs = detection_indice_pairs + [[1, 0]] * (
                topk - len(detection_indice_pairs)
            )

        detections.append(detection_indice_pairs)
    # # DEBUG
    # max_num_pair = max([len(p) for p in detections])
    # for i in range(len(detections)):
    #     if len(detections[i])<max_num_pair:
    #         detections[i] = detections[i] + [[1, 0]] * (
    #             max_num_pair - len(detections[i])
    #         )
    
    detections = torch.tensor(
        np.array(detections, dtype=np.int64).reshape(len(detections), -1),
        dtype=torch.long,
        device=outputs.device,
    )

    return detections


def _pick_phase(
    outputs: torch.Tensor,
    prob_threshold: float,
    min_peak_dist: int,
    topk: int,
    padding_value: int,
) -> torch.Tensor:
    """Pick phases of one batch.

    Args:
        outputs (torch.Tensor): 2d-Tensor.
        prob_threshold (float): Minimum probability.
        min_peak_dist (int): Minimum peak distance.
        topk (int): Only pick the phases whose probabilities are in the top k.
        padding_value (int): Ensure that each sample returns the same length.

    Returns:
        torch.Tensor: shape (N,topk)
    """
    phases = []
    for output in outputs.detach().cpu().numpy():
        samps = _detect_peaks(output, mph=prob_threshold, mpd=min_peak_dist, topk=topk)
        p_arr = np.ones(topk, dtype=np.int64) * padding_value
        p_arr[: samps.shape[0]] = samps
        phases.append(p_arr)

    phases = torch.tensor(
        np.array(phases, dtype=np.int64),
        dtype=torch.long,
        device=outputs.device,
    )

    return phases


def process_outputs(
    args: argparse.Namespace,
    outputs: Union[Tuple[torch.Tensor], torch.Tensor],
    label_names: List[str],
    sampling_rate: int,
) -> Dict[str, torch.Tensor]:
    """Process outputs of model.

    Args:
        args (argparse.Namespace): Arguments.
        outputs (Union[Tuple[torch.Tensor], torch.Tensor]): Output of model.
        sampling_rate (int): Sampling rate of waveform.
        device (torch.device): Device.

    Returns:
        Dict[str, torch.Tensor]: results like {<task>: Tensor, ...}
    """

    if not isinstance(outputs, (tuple, list)):
        outputs_list = [outputs]
    else:
        outputs_list = outputs
    results = {}
    for outputs, label_group in zip(outputs_list, label_names):
        if isinstance(label_group, (tuple, list)):
            for i, name in enumerate(label_group):
                if name in ["ppk", "spk"]:
                    phases = _pick_phase(
                        outputs=outputs[:, i],
                        prob_threshold=(
                            args.ppk_threshold if name == "ppk" else args.spk_threshold
                        ),
                        min_peak_dist=int(args.min_peak_dist * sampling_rate),
                        topk=args.max_detect_event_num,
                        padding_value=int(-1e7),
                    )
                    results[name] = phases
                elif name == "det":
                    detections = _detect_event(
                        outputs=outputs[:, i],
                        prob_threshold=args.det_threshold,
                        topk=args.max_detect_event_num,
                    )
                    results[name] = detections
                else:
                    tmp = outputs[:, i]
                    if tmp.dim() < 2:
                        tmp = tmp.unsqueeze(-1)
                    results[name] = tmp

        else:
            label_name = label_group
            results[label_name] = outputs

    return results


class ResultSaver:
    def __init__(self, item_names: list):
        self._item_names = item_names
        self._results_dict = defaultdict(list)

    def _convert_type(self, v: Union[list, torch.Tensor]) -> list:
        if isinstance(v, torch.Tensor):
            v = v.tolist()

        if isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i],list):
                    if len(v[i])==0:
                        v[i]==""
                    elif len(v[i])==1:
                        v[i] = v[i].pop()
                    else:
                        v[i] = [str(x) for x in v[i]]
                        v[i] = ",".join(v[i])
        else:
            raise TypeError(f"Unknown data type: {type(v)}")
        return v

    def _process_item(self, k: str, v: Union[list, torch.Tensor],prefix:str="") -> Tuple[str, list]:
        # One-hot -> Ind
        if Config.get_type(k) == "onehot":
            v = torch.argmax(v, dim=-1)

        # Remove padding
        if k in ["ppk", "spk"]:
            v = v.tolist()
            for i in range(len(v)):
                v[i] = [x for x in v[i] if x > 0]

        save_k = f"{prefix}{k}"

        return save_k, v

    def append(self, batch_meta_data: dict,targets:dict, results: dict) -> None:
        """Append rows.

        Args:
            batch_meta_data (dict): {col0:[row0,row1,...], col1: ...}
            targets(dict): Targets for computing metrics.
            results (dict): Results from `process_outputs`
        """
        assert isinstance(batch_meta_data, dict), f"{type(batch_meta_data)}"

        unknown_names = (set(results)|set(targets)) - set(self._item_names)
        not_found_names = set(self._item_names) - (set(results)|set(targets))

        if len(unknown_names) and not hasattr(self,"unknown_warning_flag"):
            logger.warning(f"[ResultSaver]unknown names in outputs: {unknown_names}, expected:{self._item_names} targets:{list(targets)} results:{list(results)}")
            setattr(self,"unknown_warning_flag",1)

        if len(not_found_names) > 0:
            logger.warning(f"[ResultSaver]not found names: {not_found_names}, expected:{self._item_names} targets:{list(targets)} results:{list(results)}")
            raise AttributeError(f"[ResultSaver]not found names: {not_found_names}, expected:{self._item_names} targets:{list(targets)} results:{list(results)}")


        tgt = {k: targets[k] for k in self._item_names}
        res = {k: results[k] for k in self._item_names}

        for k, v in batch_meta_data.items():
            v = self._convert_type(v)
            self._results_dict[k].extend(v)

        for k in self._item_names:

            pred_k, pred_v = self._process_item(k, res[k],prefix="pred_")
            pred_v = self._convert_type(pred_v)
            self._results_dict[pred_k].extend(pred_v)

            tgt_k, tgt_v = self._process_item(k, tgt[k],prefix="tgt_")
            tgt_v = self._convert_type(tgt_v)
            self._results_dict[tgt_k].extend(tgt_v)
            

    def save_as_csv(self, path: str) -> None:
        df = pd.DataFrame(self._results_dict)

        sdir, sname = os.path.split(path)
        if not os.path.exists(sdir):
            os.makedirs(sname)

        df.to_csv(path)
