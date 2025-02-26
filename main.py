import argparse
import warnings
import torch
import os
from utils import *
from training import *
from config import Config

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Model training/testing arguments")

    def bool_(x):
        return False if str(x).strip().lower() in ("0", "false", "f", "no", "n") else bool(x)

    # Mode
    parser.add_argument("--mode", type=str, default="train_test", metavar="MODE",
                        help="train/test/train_test (default:'train_test')")

    # Model
    parser.add_argument("--model-name", default="seist_m_dpk", type=str, metavar="MODEL_NAME",
                        help="model name: 'phasenet/eqtransformer/magnet/ditingmotion/seist' (default: seist_l_dpk)")
    parser.add_argument("--checkpoint", default="", type=str, metavar="CHECKPOINT",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--use-torch-compile", type=bool_, default=True, metavar="USE_TORCH_COMPILE",
                        help="if `True`, `torch.compile` will be called before training (default:True)")
    parser.add_argument('--llm', type=str, default="GPT2", help="the name of LLM")
    parser.add_argument('--pretrain', type=bool, default=True, help="if `True`, load pretrain LLM weight")
    parser.add_argument('--freeze', type=bool, default=True, help="if `True`, freeze most of the pretrain LLM weight")
    parser.add_argument('--LLM_start_layer', type=int, default=0, help="the used first layer index (1st layer is 0) of LLM")
    parser.add_argument('--LLM_layer_num', type=int, default=3, help="using LLM layers number")
    parser.add_argument('--d_model', type=int, default=768, help="hideen dimension of LLM")
    # parser.add_argument('--patch_size', type=int, default=16, help="patch size of data before feeding to LLM")

    # Random seed
    parser.add_argument("--seed", default=0, type=int, metavar="SEED",
                        help="random seed for everything (default:0)")

    # Logs
    parser.add_argument("--log-base", default="./logs", type=str, metavar="LOG_DIR",
                        help="path to save logs (default: './logs')")
    parser.add_argument("--log-step", default=4, type=int, metavar="log_step",
                        help="print metrics every log_step steps (default: 4)")
    parser.add_argument("--use-tensorboard", default=True, type=bool_, metavar="USE_TENSORBOARD",
                        help="whether to use tensorboard (default: True)")
    
    # Save results
    parser.add_argument("--save-test-results", default=True, type=bool_, metavar="SAVE_TEST_RESULTS",
                        help="whether to save test restuls (default: True)")

    # Distributed training
    parser.add_argument("--find-unused-parameters", type=bool_, default=False, metavar="FUP",
                        help="argument of `torch.nn.parallel.DistributedDataParallel` (default:False)")

    # Single GPU
    parser.add_argument("--device", type=str, default="cuda:0", metavar="DEVICE",
                        help="device. If distributed mode is initialized, this argument will be ignored. (default:'cuda:0')")

    # Dataset
    parser.add_argument("--data", default="/root/data/Datasets/Diting50hz", metavar="DATA", type=str,
                        help="path to dataset")
    parser.add_argument("--dataset-name", default="diting_light", type=str, metavar="DATASET_NAME",
                        help="name of dataset ('diting', 'diting_light', 'pnw', 'pnw_light' or 'sos') (default: 'diting_light')")
    parser.add_argument("--data-split", type=bool_, default=True, metavar="DATA_SPLIT",
                        help="whether split dataset to train/val/test (default:True)")
    parser.add_argument("--train-size", type=float, default=0.8, metavar="TRAIN_SIZE",
                        help="size of train set (default:0.8)")
    parser.add_argument("--val-size", type=float, default=0.1, metavar="VAL_SIZE",
                        help="size of val set (default:0.1)")

    # Data loader
    parser.add_argument("--shuffle", type=bool_, default=True, metavar="SHUFFLE",
                        help="whether shuffle data. (default:True)")
    parser.add_argument("--workers", default=8, type=int, metavar="WORKERS",
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--pin-memory", default=True, type=bool_, metavar="PM",
                        help="pin memory (default: True)")

    # Data preprocess
    parser.add_argument("--in-samples", default=8192, type=int, metavar="IN_SAMPLES",
                        help="the length of input data (default: 8192)")
    parser.add_argument("--label-width", type=float, default=0.5, metavar="LABEL_WIDTH",
                        help="width of soft-label (in seconds) (default:0.5)")
    parser.add_argument("--label-shape", type=str, default="gaussian", metavar="LABEL_SHAPE",
                        help="shape of soft-label ('gaussian' 'triangle' 'box' or 'sigmoid') (default: gaussian)")
    parser.add_argument("--coda-ratio", default=2.0, type=float, metavar="CODA_RATIO",
                        help="coda ratio (default:2)")
    parser.add_argument("--norm-mode", default="std", type=str, metavar="NORM_MODE",
                        help="mode of normalization ('max','std' or '') (default: 'std')")
    parser.add_argument("--min-snr", type=float, default=-float("inf"), metavar="MIN_SNR",
                        help="waveform will be regarded as noise if `all(snr)<min_snr` (default:-inf)")
    parser.add_argument("--p-position-ratio", type=float, default=-1, metavar="P_POSITION_RATIO",
                        help="The position of phase-p in the waveform. Only takes effect when `0 <= p_position_ratio <= 1` (default: -1)")

    # Data augmentation
    parser.add_argument("--augmentation", type=bool_, default=True, metavar="AUGMENTATION",
                        help="whether use data augmentation. (default:True)")
    parser.add_argument("--add-event-rate", default=0.0, type=float, metavar="ADD_EV_RATE",
                        help="Add event rate (default:0.0)")
    parser.add_argument("--max-event-num", default=1, type=int, metavar="MAX_EV_NUM",
                        help="max number of event (default:1)")
    parser.add_argument("--shift-event-rate", default=0.2, type=float, metavar="SHIFT_EV_RATE",
                        help="shift event rate (default:0.2)")
    parser.add_argument("--add-noise-rate", default=0.4, type=float, metavar="ADD_NOISE_RATE",
                        help="add noise rate (default:0.4)")
    parser.add_argument("--add-gap-rate", default=0.4, type=float, metavar="ADD_GAP_RATE",
                        help="add gap rate (default:0.4)")
    parser.add_argument("--min-event-gap", default=0.5, type=float, metavar="MIN_EV_GAP",
                        help="minimum event gap (in seconds) (default:0.5)")
    parser.add_argument("--drop-channel-rate", default=0.4, type=float, metavar="DROP_CH_RATE",
                        help="drop channel rate (default:0.4)")
    parser.add_argument("--scale-amplitude-rate", default=0.4, type=float, metavar="SCALE_AMP_RATE",
                        help="scale amplitude rate (default:0.4)")
    parser.add_argument("--pre-emphasis-rate", default=0.4, type=float, metavar="PRE_EMPH_RATE",
                        help="pre-emphaseis rate (default:0.4)")
    parser.add_argument("--pre-emphasis-ratio", default=0.97, type=float, metavar="PRE_EMPH_RATIO",
                        help="pre-emphasis ratio (default:0.97)")
    parser.add_argument("--generate-noise-rate", default=0.05, type=float, metavar="GEN_NOISE_RATE",
                        help="generate noise rate (default:0.05)")
    parser.add_argument("--mask-percent", default=0, type=int, metavar="MASK_PERCENT",
                        help="the percentage of the total mask window size to the entire waveform length,"
                        " where the window size is 0.5s (range:0-100) (default: 0)")
    parser.add_argument("--noise-percent", default=0, type=int, metavar="NOISE_PERCENT",
                        help="the percentage of the total noise window size to the entire waveform length,"
                        " where the window size is 0.5s (range:0-100) (default: 0)")

    # Train
    parser.add_argument("--epochs", default=200, type=int, metavar="EPOCHS",
                        help="number of total epochs (default: 200)")
    parser.add_argument("--patience", default=30, type=int, metavar="PATIENCE",
                        help="how many epochs to wait before stopping when loss is not improving (default: 30)")
    parser.add_argument("--steps", default=0, type=int, metavar="STEPS",
                        help="number of total steps. if `steps > 0`, `epochs` will be ignored. (default: 0)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="START_EPOCH",
                        help="manual epoch number (useful on restarts) (default: 0)")
    parser.add_argument("--batch-size", default=500, type=int, metavar="BATCH_SIZE",
                        help="batch size (default: 500), this is the batch size of each worker (process)")
    parser.add_argument("--optim", default="Adam", type=str, metavar="OPTIM",
                        help="name of optimizer (default: 'Adam')")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="MOMENTUM",
                        help="momentum of optimizer SGD (default: 0.9)")
    parser.add_argument("--weight_decay", default=0.0, type=float, metavar="WEIGHT_DECAY",
                        help="weight_decay of optimizer (default: 0.)")
    parser.add_argument("--use-lr-scheduler", default=True, type=bool_, metavar="USE_LR_SCHEDULER",
                        help="whether use lr_scheduler (default: True)")
    parser.add_argument("--lr-scheduler-mode", default="exp_range", metavar="LR_SCHEDULER_MODE", type=str,
                        help="one of {'triangular', 'triangular2', 'exp_range'} (default: 'exp_range')")
    parser.add_argument("--base-lr", default=8e-5, type=float, metavar="BASE_LR",
                        help="minimum learning rate (default: 5e-5)")
    parser.add_argument("--max-lr", default=1e-3, type=float, metavar="MAX_LR",
                        help="maximum learning rate (default: 1e-3)")
    parser.add_argument("--warmup-steps", default=2000, type=float, metavar="WARMUP_STEPS",
                        help="number of training iterations in the increasing half of a cycle."
                        " If `0 < warmup_steps < 1`, it will be treated as a ratio of total steps. (default: 1000)")
    parser.add_argument("--down-steps", default=3000, type=float, metavar="DOWN_STEPS",
                        help="number of training iterations in the decreasing half of a cycle."
                        " If `0 < down_steps < 1`, it will be treated as a ratio of total steps."
                        " If `down_steps == 0`, it will be set to `steps - warmup_steps`(default: 1000)")

    # Val/Test
    parser.add_argument("--time-threshold", default=0.1, type=float, metavar="TIME_THRESHOLD",
                        help="Residual threshold (in seconds) (default: 0.5)")
    parser.add_argument("--min-peak-dist", default=1.0, type=float, metavar="MIN_PEAK_DIST",
                        help="Detect peaks that are at least separated by minimum peak distance (in seconds) (defult: 1.0)")
    parser.add_argument("--ppk-threshold", default=0.3, type=float, metavar="PPK_THRESHOLD",
                        help="Probability threshold of phase-P PicKing (default: 0.3)")
    parser.add_argument("--spk-threshold", default=0.3, type=float, metavar="SPK_THRESHOLD",
                        help="Probability threshold of phase-S PicKing (default: 0.3)")
    parser.add_argument("--det-threshold", default=0.5, type=float, metavar="DET_THRESHOLD",
                        help="Probability threshold of DETection (default: 0.5)")
    parser.add_argument("--max-detect-event-num", default=1, type=int, metavar="MAX_DETECT_EV_NUM",
                        help="max number of detected events (default: 1)")

    args = parser.parse_args()


    if not 0<=args.p_position_ratio<=1:
        args.p_position_ratio = -1
    else:
        print(f"P position ratio: {args.p_position_ratio}")

    args.log_base = os.path.abspath(args.log_base)
    args.data = os.path.abspath(args.data)

    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)

    return args


def main_worker(args, device):

    log_dir = (
        os.path.join(args.log_base, f"{get_time_str()}_{args.model_name}_{args.dataset_name}")
        if not args.checkpoint
        else args.checkpoint.split("checkpoints")[0]
    )
    logger.set_logdir(log_dir)
    logger.set_logger("global")
    
    if is_main_process():
        # logger.info(f"device: {device}")
        # logger.info(f"pid: {os.getpid()}")
        logger.info(f"\n{strfargs(args, Config)}")
    
    mode = args.mode.split("_")
    if "train" in mode:
        setup_seed(args.seed)
        ckpt_path = train_worker(args,device)
        args.checkpoint = ckpt_path

    if "test" in mode:
        setup_seed(args.seed)
        test_worker(args,device)

    if not (set(("train", "test")) & set(mode)):
        raise ValueError(
            f"`mode` must be 'train','test' or 'train_test', got '{args.mode}'"
        )


if __name__ == "__main__":

    args = get_args()

    args.distributed = init_distributed_mode()

    if args.distributed:
        args.device = f"cuda:{get_local_rank()}"

    device = torch.device(args.device)

    if args.use_torch_compile and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    main_worker(args, device)
