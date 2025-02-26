import torch
import h5py
import numpy as np
import pandas as pd
from models import create_model, load_checkpoint
from utils import vis_phase_picking


def normalize(data: np.ndarray, mode: str):
    data -= np.mean(data, axis=1, keepdims=True)
    if mode == "max":
        max_data = np.max(data, axis=1, keepdims=True)
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == "std":
        std_data = np.std(data, axis=1, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    elif mode == "":
        return data
    else:
        raise ValueError(f"Supported mode: 'max','std', got '{mode}'")
    return data


def load_data(
    data_path: str = "/root/data/Datasets/Diting50hz/DiTing330km_part_0.hdf5",
    trace_name: str = "000014.0100",
):
    # Read HDF5
    with h5py.File(data_path, "r") as f:
        data = f.get(f"earthquake/{trace_name}")
        data = np.array(data).astype(np.float32).T
    
    return data


def load_model(
    model_name: str,
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 3,
    in_samples: int = 8192,  # Only 'EQTransformer' and 'BAZ-Network' need this argument.
):
    # Model init
    model = create_model(
        model_name=model_name, in_channels=in_channels, in_samples=in_samples
    )

    # Load parameters
    ckpt = load_checkpoint(ckpt_path, device=device)
    model_state_dict = ckpt["model_dict"] if "model_dict" in ckpt else ckpt
    model.load_state_dict(model_state_dict)
    model.to(device)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step.1 - Load waveforms and labels
    data_dir = "/ailab/user/wangxinghao/project/datasets/DiTing330km/"
    key = "397372.0103"

    meta = pd.read_csv(data_dir + "DiTing330km_fewshot.csv")
    event = meta[meta["key"].astype(str) == key]
    part = event["part"].values[0]
    phases = [event["p_pick"].values, event["s_pick"].values]

    waveform_ndarray = load_data(
        data_path=data_dir + f"DiTing330km_part_{part}.hdf5",
        trace_name=key,
    )
    waveform_ndarray = waveform_ndarray[:, :8192]
    waveform_ndarray = normalize(waveform_ndarray, mode="std")
    waveform_tensor = torch.from_numpy(waveform_ndarray).reshape(1, 3, -1).to(device)

    # Step.2 - Load Model 
    model = load_model(
        model_name="SeisGPT_dpk",
        ckpt_path="./results/2024-11-05_17-04-07_SeisGPT_LoraAll_dpk_diting_light***/checkpoints/model-102.pth",
        device=device,
        in_channels=3,
    )


    # Step.3 - Inference
    preds_tensor = model(waveform_tensor)
    preds_ndarray = preds_tensor.detach().cpu().numpy().reshape(3, -1)


    # Step.4 - Visualization 
    vis_phase_picking(
        waveforms=waveform_ndarray,
        waveforms_labels=["Z", "N", "E"],
        preds=preds_ndarray,
        true_phase_idxs=phases,
        true_phase_labels=["P", "S"],
        pred_phase_labels=["$\hat{D}$", "$\hat{P}$", "$\hat{S}$"],
        sampling_rate=None,
        save_name=f"_{key}",
        save_dir="./dpk_result_figures/",
        formats=["png"],
    )

    
    """Available Models:
    ['eqtransformer', 'phasenet', 'magnet', 'baz_network', 'distpt_network'(code only), 'ditingmotion', 
    'seist_s_dpk', 'seist_m_dpk', 'seist_l_dpk', 'seist_s_pmp', 'seist_m_pmp', 'seist_l_pmp', 
    'seist_s_emg', 'seist_m_emg', 'seist_l_emg', 'seist_s_baz', 'seist_m_baz', 'seist_l_baz']"""