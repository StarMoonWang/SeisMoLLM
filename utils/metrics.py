import torch
import torch.distributed as dist
import numpy as np
import math
from .misc import reduce_tensor, gather_tensors_to_list
from typing import Tuple
import copy
from typing import List, Dict, Union




class Metrics:
    """Compute metrics (batch-wise average).

    Available metrics: `Precision`, `Recall`, `F1`, `Mean`, `Std`, `MAE`, `MAPE`, `R2`
    """

    _epsilon = 1e-6
    _avl_regr_keys = ("sum_res", "sum_squ_res", "sum_abs_res", "sum_abs_per_res")
    _avl_cmat_keys = ("tp", "predp", "possp")
    _avl_metrics = ("precision", "recall", "f1", "mean", "std", "mae", "mape", "r2")


    def __init__(
        self,
        task: str,
        metric_names: Union[list, tuple],
        sampling_rate: int,
        time_threshold: int,
        num_samples: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            task: str
                Task name. See :class:`SeisT.config.Config` for more details.
            metric_names:Union[list,tuple]
                Names of metrics.
            sampling_rate:int
                Sampling rate of waveform.
            time_threshold:int
                Threshold for phase-picking.
            num_samples:int
                Number of samples of waveform.
            device: torch.device
                Device.
        """

        self.device = device

        self._t_thres = int(time_threshold * sampling_rate)

        self._task = task.lower()
        self._metric_names = tuple(n.lower() for n in metric_names)

        self._num_samples = num_samples

        unexpected_keys = set(self._metric_names) - set(self._avl_metrics)
        assert set(self._metric_names).issubset(
            self._avl_metrics
        ), f"Unexpected metrics:{unexpected_keys}"

        data_keys = self._metric_names
        if set(self._metric_names) & set(("precision", "recall", "f1")):
            data_keys += self._avl_cmat_keys
        if set(self._metric_names) & set(("mean", "std", "mae", "mape")):
            data_keys += self._avl_regr_keys

        self._data={
            k: torch.tensor(0, dtype=torch.float32, device=self.device)
            for k in data_keys
        }
        self._data["data_size"] = torch.tensor(0, dtype=torch.long, device=self.device)
        
        self._tgts: torch.Tensor = None

        self._results: Dict[str,float] = {}
        
        self._modified = True
        

    def synchronize_between_processes(self):
        """
        Synchronize metrics between processes
        """
        dist.barrier()

        for k in self._data:
            self._data[k] = reduce_tensor(self._data[k])

        if isinstance(self._tgts, torch.Tensor):
            tgts_list = gather_tensors_to_list(self._tgts)
            self._tgts = torch.cat(tgts_list, dim=0)

        dist.barrier()

        self._modified = True


    def _order_phases(
        self, targets: torch.Tensor, preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match the order of predictions and labels"""

        num_phases = targets.size(-1)
        _targets = targets.clone().detach().cpu().numpy()
        _preds = preds.clone().detach().cpu().numpy()

        for i, (target_i, pred_i) in enumerate(zip(_targets, _preds)):
            orderd = np.zeros_like(pred_i)
            dmat = np.abs(
                target_i[:, np.newaxis].repeat(num_phases, axis=1)
                - pred_i[np.newaxis, :].repeat(num_phases, axis=0)
            )
            for _ in range(num_phases):
                ind = dmat.argmin()
                ito, ifr = ind // num_phases, ind % num_phases
                orderd[ito] = pred_i[ifr]
                dmat[ito, :] = int(1 / self._epsilon)
                dmat[:, ifr] = int(1 / self._epsilon)
            _preds[i] = orderd

        preds.copy_(torch.from_numpy(_preds))
        return targets, preds

    @torch.no_grad()
    def compute(
        self, targets: torch.Tensor, preds: torch.Tensor, reduce: bool = False
    ) -> None:
        """
        Args:
            targets: torch.Tensor
                Labels. Shape: (N, L), (N, Classes), (N, 2 * D) or (N, 1)
            preds: torch.Tensor
                Predictions. Shape: (N, L), (N, Classes), (N, 2 * D) or (N, 1)
            reduce: bool
                For distributed training.
        """
        assert targets.size(0) == preds.size(0), f"`{targets.size()}` != `{preds.size()}`"
        assert targets.dim() == 2, f"dim:{targets.dim()}, shape:{targets.size()}"

        self._data["data_size"] += targets.size(0)

        targets = targets.clone().detach().to(self.device)
        preds = preds.clone().detach().to(self.device)
        mask = 1.0

        if set(self._metric_names) & set(("precision", "recall", "f1")):
            if self._task in ["ppk", "spk"]:
                targets = targets.type(torch.long)
                preds = preds.type(torch.long)
                if targets.size(-1) > 1:
                    targets, preds = self._order_phases(targets, preds)

                preds_bin = (preds >= 0) & (preds < self._num_samples)
                targets_bin = (targets >= 0) & (targets < self._num_samples)
                
                ae = torch.abs(targets - preds)
                mask = tp_bin = preds_bin & targets_bin & (ae <= self._t_thres)

                self._data["tp"] = torch.sum(tp_bin)
                self._data["predp"] = torch.sum(preds_bin)
                self._data["possp"] = torch.sum(targets_bin)

            elif self._task in ["det"]:
                targets = targets.type(torch.long)
                preds = preds.type(torch.long)
                
                bs = targets.size(0)
                
                targets = targets.reshape(bs,-1,2)
                preds = preds.reshape(bs,-1,2)

                indices = torch.arange(self._num_samples,device=self.device).unsqueeze(0).unsqueeze(0)
                
                targets_bin = torch.sum((targets[:,:,:1] <= indices) & (indices <=targets[:,:,1:]),dim=-2)
                preds_bin = torch.sum((preds[:,:,:1] <= indices) & (indices <=preds[:,:,1:]),dim=-2)

                self._data["tp"] = torch.sum(
                    torch.round(torch.clip(targets_bin * preds_bin, 0, 1))
                )
                self._data["predp"] = torch.sum(
                    torch.round(torch.clip(preds_bin, 0, 1))
                )
                self._data["possp"] = torch.sum(
                    torch.round(torch.clip(targets_bin, 0, 1))
                )

            else:
                assert (
                    targets.size() == preds.size()
                ), f"`{targets.size()}` != `{preds.size()}`"
                assert targets.size(-1) > 1, f"The input must be one-hot."

                # Scatter is faster than fancy indexing.
                preds_indices = preds.topk(1).indices
                preds = preds.zero_().scatter_(dim=1,index=preds_indices,value=1)
                
                targets_indices = targets.topk(1).indices
                targets = targets.zero_().scatter_(dim=1,index=targets_indices,value=1)

                self._data["tp"] = torch.sum(targets * preds, dim=0)
                self._data["predp"] = torch.sum(preds, dim=0)
                self._data["possp"] = torch.sum(targets, dim=0)

        if set(self._metric_names) & set(("mean", "std", "mae", "mape", "r2")):
            res = targets - preds
            # BAZ
            if self._task in ["baz"]:
                res = torch.where(
                    res.abs() > 180, -torch.sign(res) * (360 - res.abs()), res
                )

            if "mean" in self._metric_names:
                self._data["sum_res"] = (res * mask).type(torch.float32).mean(-1).sum()

            if "std" in self._metric_names:
                self._data["sum_squ_res"] = (
                    torch.pow(res * mask, 2).type(torch.float32).mean(-1).sum()
                )

            if "mae" in self._metric_names:
                self._data["sum_abs_res"] = (
                    (res * mask).abs().type(torch.float32).mean(-1).sum()
                )

            if "mape" in self._metric_names:
                self._data["sum_abs_per_res"] = (
                    (res * mask / (targets + self._epsilon))
                    .abs()
                    .type(torch.float32)
                    .mean(-1)
                    .sum()
                )

            if "r2" in self._metric_names:
                self._tgts = targets
                if "sum_squ_res" not in self._data:
                    self._data["sum_squ_res"] = (
                        torch.pow(res * mask, 2).type(torch.float32).mean(-1).sum()
                    )

        if reduce:
            self.synchronize_between_processes()

        self._modified = True

    def add(self, b) -> None:
        if not type(self) == type(b):
            raise TypeError(f"Type of `b` must be `Metrics`, got `{type(b)}`")

        if (set(self._data) | set(b._data)) - (set(self._data) & set(b._data)):
            raise TypeError(
                f"Mismatched data fields: `{set(self._data)}` and `{set(b._data)}`"
            )

        for k in self._data:
            self._data[k] = self._data[k] + b._data[k]

        tgts_to_cat = list(
            filter(lambda x: isinstance(x, torch.Tensor), [self._tgts, b._tgts])
        )
        if tgts_to_cat:
            self._tgts = torch.cat(tgts_to_cat, dim=0)

        self._modified = True

    def __add__(a, b):
        if not type(a) == type(b):
            raise TypeError(
                f"Unsupported operand type(s) for `+`: `{type(a)}` and `{type(b)}`"
            )

        if (set(a._data) | set(b._data)) - (set(a._data) & set(b._data)):
            raise TypeError(
                f"Mismatched data fields: `{set(a._data)}` and `{set(b._data)}`"
            )

        c = copy.deepcopy(a)
        for k in c._data:
            c._data[k] = a._data[k] + b._data[k]

        tgts_to_cat = list(
            filter(lambda x: isinstance(x, torch.Tensor), [a._tgts, b._tgts])
        )
        if tgts_to_cat:
            c._tgts = torch.cat(tgts_to_cat, dim=0)
        c._modified = True

        return c

    def _update_metric(self, key: str) -> torch.Tensor:
        """Update value of metric."""

        if key == "precision":
            v = self._data["precision"] = (
                self._data["tp"] / (self._data["predp"] + self._epsilon)
            ).mean()
        elif key == "recall":
            v = self._data["recall"] = (
                self._data["tp"] / (self._data["possp"] + self._epsilon)
            ).mean()
        elif key == "f1":
            pr = self._data["tp"] / (self._data["predp"] + self._epsilon)
            re = self._data["tp"] / (self._data["possp"] + self._epsilon)
            v = self._data["f1"] = (2 * pr * re / (pr + re + self._epsilon)).mean()
        elif key == "mean":
            v = self._data["mean"] = self._data["sum_res"] / self._data["data_size"]
        elif key == "std":
            v = self._data["std"] = torch.sqrt(
                self._data["sum_squ_res"] / self._data["data_size"]
            )
        elif key == "mae":
            v = self._data["mae"] = self._data["sum_abs_res"] / self._data["data_size"]
        elif key == "mape":
            v = self._data["mape"] = (
                self._data["sum_abs_per_res"] / self._data["data_size"]
            )
        elif key == "r2":
            t = self._tgts - self._tgts.mean()
            # BAZ
            if self._task in ["baz"]:
                t = torch.where(t.abs() > 180, -torch.sign(t) * (360 - t.abs()), t)
            v = 1 - (
                self._data["sum_squ_res"]
                / (torch.pow(t, 2).mean(-1).sum() + self._epsilon)
            )
        else:
            raise ValueError(f"Unexpected key name: '{key}'")

        return v

    def _update_all_metrics(self) -> dict:
        if self._modified or len(self._results)==0:
            self._results = {
                k: self._update_metric(k).item() for k in self._metric_names
            }
            self._modified = False
        return self._results

    def get_metric(self, name: str) -> float:
        self._update_all_metrics()
        return self._results[name]

    def get_metrics(self, names: List[str]) -> Dict[str, float]:
        self._update_all_metrics()
        metrics_dict = {}
        for name in names:
            name_lower = name.lower()
            if name_lower in self._avl_metrics:
                metrics_dict[name] = self.get_metric(name_lower)
        return metrics_dict

    def metric_names(self) -> List[str]:
        return list(self._metric_names)

    def get_all_metrics(self) -> Dict[str, float]:
        return self._update_all_metrics()

    def __repr__(self) -> str:
        entries = [
            f"{k.upper()} {v:6.4f}" for k, v in self._update_all_metrics().items()
        ]
        string = "  ".join(entries)
        return string

    def to_dict(self) -> dict:
        self._update_all_metrics()
        metrics_dict = {}
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                v = v.item() if v.dim() == 0 else v.tolist()

            if isinstance(v, (list, tuple, np.ndarray)):
                for i, vi in enumerate(v):
                    if isinstance(vi, torch.Tensor):
                        vi = vi.item()
                    metrics_dict[f"{k}.{i}"] = vi
            else:
                metrics_dict[k] = v
        return metrics_dict

