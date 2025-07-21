from collections import defaultdict
import math
import re
from functools import partial
from typing import Any
import torch
from models import (
    BCELoss,
    BinaryFocalLoss,
    CELoss,
    ConbinationLoss,
    FocalLoss,
    HuberLoss,
    MousaviLoss,
    MSELoss,
    get_model_list,
)


class Config:
    #  ////////////////////////////////////////////////////////////////////////////// Models
    _model_conf_keys = (
        "loss",
        "labels",
        "eval",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )
    
    """
    Model configuration fields:

        - The keys of the dict are the model names, and regular expressions are supported.

        - The value of `'loss'` is the class of loss module.
        All the loss modules are defined in 'models/loss.py'.
        Use `functools.partial` if there are arguments that need to be passed to the module class.

        - The values of `'inputs'` are used for generating inputs of the model.
        If multiple inputs are placed in a `tuple` or `list`, they will be concatenated
        into one tensor, such as `[("z", "n", "e"), "ppks", ... ]`.
        See `Config._avl_io_items` for all available inputs.

        - The values of `'labels'` are used for computing loss value.
        If multiple labels are placed in a `tuple` or `list`, they will be concatenated
        into one tensor, such as `[("det", "ppk", "spk"), "pmp", ... ]`.
        Note: the order of `'labels'` must be the same as the model's outputs.
        See `Config._avl_io_items` for all available inputs.

        - The values of `eval` are used for calculating metrics. 
        All items in this list will be evaluated

        - `'targets_transform_for_loss'` is used to transform data from data reader.
        It will be called before computing loss value.

        - `'outputs_transform_for_loss'` is used to transform output data from model.
        It will be called before computing loss value.

        - `'outputs_transform_for_results'`  is used to transform output data from model.
        It will be called before post-processing.

    """

    models = {

        # --------------------------------------------------------- PhaseNet
        "phasenet": {
            "loss": partial(CELoss, weight=[[1], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- EQTransformer
        "eqtransformer": {
            "loss": partial(BCELoss, weight=[[0.05], [0.4], [0.55]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- MagNet
        "magnet": {
            "loss": MousaviLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: x[:, 0].reshape(-1, 1),
        },
        # --------------------------------------------------------- BAZ Network
        "baz_network": {
            "loss": partial(ConbinationLoss, losses=[MSELoss, MSELoss]),
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": lambda x: (
                (x * math.pi / 180).cos(),
                (x * math.pi / 180).sin(),
            ),
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: torch.atan2(x[1], x[0])
            * 180
            / math.pi,
        },
        # --------------------------------------------------------- dist-PT Network
        # The model has not been reproduced due to no travel time data in the DiTing dataset.
        #
        "distpt_network": {
            "loss": partial(ConbinationLoss, losses=[MousaviLoss,MousaviLoss], losses_weights=[0.1,0.9]),
            "inputs": [["z", "n", "e"]],
            "labels": ["dis", "ptt"],
            "eval": ["dis", "ptt"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: (x[0][:, 0].reshape(-1, 1), x[1][:, 0].reshape(-1, 1)),
        },
        # --------------------------------------------------------- DiTingMotion
        "ditingmotion": {
            "loss": partial(ConbinationLoss, losses=[FocalLoss, FocalLoss]),
            "inputs": [["z", "dz"]],
            "labels": ["clr", "pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda xs: [x.softmax(-1) for x in xs],
        },
        # --------------------------------------------------------- SeisT (Detection, Picking)
        "seist_.*?_dpk.*": {
            "loss": partial(BCELoss, weight=[[0.5], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (P-motion-polarity)
        "seist_.*?_pmp": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z","n","e"]],
            "labels": ["pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Magnitude)
        "seist_.*?_emg": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Azimuth)
        "seist_.*?_baz": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisT (Distance)
        "seist_.*?_dis": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["dis"],
            "eval": ["dis"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },


        # --------------------------------------------------------- SeisMoLLM (Phase Picking)
        "SeisMoLLM_dpk.*": {
            "loss": partial(BCELoss, weight=[[0], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            # incorporating detection task instead degrades performance
            # "labels": [["det", "ppk", "spk"]],
            # "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisMoLLM (P-motion-polarity)
        "SeisMoLLM_pmp": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z","n","e"]],
            "labels": ["pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisMoLLM (Magnitude)
        "SeisMoLLM_emg": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # --------------------------------------------------------- SeisMoLLM (Back-Azimuth)
        "SeisMoLLM_baz": {
            "loss": partial(ConbinationLoss, losses=[HuberLoss, HuberLoss]),
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": lambda x: (
                (x * math.pi / 180).cos(),
                (x * math.pi / 180).sin(),
            ),
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: torch.atan2(x[1], x[0])
            * 180
            / math.pi,
        },
        # --------------------------------------------------------- SeisMoLLM (Distance)
        "SeisMoLLM_dis": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["dis"],
            "eval": ["dis"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        }
        
    }

    #  ////////////////////////////////////////////////////////////////////////////// Metrics
    """
    See :class:`~SeisT.utils.metrics.Metrics` for more details.
    """
    _avl_metrics = (
        "precision",
        "recall",
        "f1",
        "mean",
        "std",
        "mae",
        "mape",
        "rmse",
        "r2",
    )

    #  ////////////////////////////////////////////////////////////////////////////// Available input and output items
    
    _avl_io_item_types = ("soft", "value", "onehot")

    _avl_io_items = {
        # -------------------------------------------------------------------------- Channel-Z
        "z": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Channel-N
        "n": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Channel-E
        "e": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(Z)
        "dz": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(N)
        "dn": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- Diff(E)
        "de": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        # -------------------------------------------------------------------------- 1-P(p)-P(s)
        "non": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(d)
        "det": {"type": "soft", "metrics": ["precision", "recall", "f1"]},
        # -------------------------------------------------------------------------- P(p)
        "ppk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape", "rmse"],
        },
        # -------------------------------------------------------------------------- P(s)
        "spk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape", "rmse"],
        },
        # -------------------------------------------------------------------------- P(p+)
        "ppk+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(s+)
        "spk+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- P(d+)
        "det+": {"type": "soft", "metrics": []},
        # -------------------------------------------------------------------------- Phase-P indices
        "ppks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "rmse", "r2"]},
        # -------------------------------------------------------------------------- Phase-S indices
        "spks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "rmse", "r2"]},
        # -------------------------------------------------------------------------- Event magnitude
        "emg": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        # -------------------------------------------------------------------------- Station magnitude
        "smg": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        # -------------------------------------------------------------------------- Back azimuth
        "baz": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        # -------------------------------------------------------------------------- Distance
        "dis": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        # -------------------------------------------------------------------------- P motion polarity
        "pmp": {
            "type": "onehot",
            "metrics": ["precision", "recall", "f1"],
            "num_classes": 2,
        },
        # -------------------------------------------------------------------------- Clarity 
        "clr": {
            "type": "onehot",
            "metrics": ["precision", "recall", "f1"],
            "num_classes": 2,
        },
        # -------------------------------------------------------------------------- P Travel Time (s)
        "ptt": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]}
    }
    #  ////////////////////////////////////////////////////////////////////////////// (DO NOT modify the following methods)

    @classmethod
    def check_and_init(cls):
        
        cls._type_to_ioitems = defaultdict(list)

        for k, v in cls._avl_io_items.items():
            cls._type_to_ioitems[v["type"]].append(k)

        # Check models
        useless_model_conf = list(cls.models)
        registered_models = get_model_list()
        for reg_model_name in registered_models:
            for re_name in cls.models:
                if re.findall(re_name, reg_model_name):
                    if re_name in useless_model_conf:
                        useless_model_conf.remove(re_name)

        if len(useless_model_conf) > 0:
            print(f"Useless configurations: {useless_model_conf}")

        # Check models' configuration
        for name, conf in cls.models.items():
            missing_keys = set(cls._model_conf_keys) - set(conf)
            if len(missing_keys) > 0:
                raise Exception(f"Model:'{name}'  Missing keys:{missing_keys}")
            expanded_labels = sum(
                [g if isinstance(g, (tuple, list)) else [g] for g in conf["labels"]], []
            )
            unknown_labels = set(expanded_labels) - set(cls._avl_io_items)
            if len(unknown_labels) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown labels:{unknown_labels}"
                )

            expanded_inputs = sum(
                [g if isinstance(g, (tuple, list)) else [g] for g in conf["inputs"]], []
            )
            unknown_inputs = set(expanded_inputs) - set(cls._avl_io_items)
            if len(unknown_inputs) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown inputs:{unknown_labels}"
                )

            unknown_tasks = set(conf["eval"]) - set(cls._avl_io_items)
            if len(unknown_tasks) > 0:
                raise NotImplementedError(
                    f"Model:'{name}'  Unknown tasks:{unknown_tasks}"
                )

        # Check io-items
        for k, v in cls._avl_io_items.items():
            if v["type"] not in cls._avl_io_item_types:
                raise NotImplementedError(f"Unknown item type: {v['type']}, item: {k}")

            unknown_metrics = set(v["metrics"]) - set(cls._avl_metrics)
            if len(unknown_metrics) > 0:
                raise NotImplementedError(
                    f"Unknown metrics:{unknown_metrics} , item: {k}"
                )

    @classmethod
    def get_io_items(cls, type: str = None) -> list:
        if type is None:
            return list(cls._avl_io_items)
        else:
            return cls._type_to_ioitems[type]
        
    @classmethod
    def get_type(cls, name: str) -> list:
        return cls._avl_io_items[name]["type"]

    @classmethod
    def get_num_classes(cls, name: str) -> int:
        if name not in cls._avl_io_items:
            raise ValueError(f"Name {name} not exists.")

        item_type = cls._avl_io_items[name]["type"]
        if item_type != "onehot":
            raise Exception(f"Type of item '{name}' is '{item_type}'.")

        num_classes = cls._avl_io_items[name]["num_classes"]

        return num_classes

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get model configuration"""

        registered_models = get_model_list()

        if model_name not in registered_models:
            raise NotImplementedError(
                f"Unknown model:'{model_name}', registered: {registered_models}"
            )

        tgt_model_conf_keys = []
        for re_name in cls.models:
            if re.findall(re_name, model_name):
                tgt_model_conf_keys.append(re_name)
        if len(tgt_model_conf_keys) < 1:
            raise Exception(f"Missing configuration of model {model_name}")
        elif len(tgt_model_conf_keys) > 1:
            raise Exception(
                f"Model {model_name} matches multiple configuration items: {tgt_model_conf_keys}"
            )
        tgt_conf_key = tgt_model_conf_keys.pop()

        conf = cls.models[tgt_conf_key]

        return conf

    @classmethod
    def get_model_config_(cls, model_name: str, *attrs) -> Any:
        """Get model configurations"""
        model_conf = cls.get_model_config(model_name=model_name)
        attrs_conf = []
        for attr_name in attrs:
            if attr_name not in model_conf:
                raise Exception(
                    f"Unknown attribute:'{attr_name}', supported: {list(model_conf)}"
                )
            attrs_conf.append(model_conf[attr_name])
        if len(attrs_conf) == 1:
            conf = attrs_conf[0]
        else:
            conf = tuple(attrs_conf)
        return conf
    
    @classmethod
    def get_num_inchannels(cls,model_name:str) ->int:
        """Get number of input channels"""
        in_channels = 0
        inps = cls.get_model_config_(model_name,"inputs")
        for inp in inps:
            if isinstance(inp,(list,tuple)):
                if cls._avl_io_items[inp[0]]["type"] == "soft":
                    in_channels = len(inp)
                    break

        if in_channels<1:
            raise Exception(f"Incorrect input channels. Model:{model_name} Inputs:{inps}")
        return in_channels
                

    @classmethod
    def get_metrics(cls, item_name: str) -> list:
        """Get metrics list"""
        if item_name not in cls._avl_io_items:
            raise Exception(
                f"Unknown item:'{item_name}', supported: {list(cls._avl_io_items)}"
            )
        metrics = cls._avl_io_items[item_name]["metrics"]
        return metrics

    @classmethod
    def get_loss(cls, model_name: str):
        """Create a loss instance.

        Args:
            model_name (str): Model name.

        Returns:
            nn.Module: Loss instance.
        """
        Loss = cls.get_model_config(model_name)["loss"]
        return Loss()


Config.check_and_init()
