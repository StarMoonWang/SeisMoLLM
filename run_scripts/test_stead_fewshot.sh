export OMP_NUM_THREADS=6

dt=`date +'%Y-%m-%d_%H-%M-%S'`

torchrun --nnodes 1 --nproc_per_node 4 main.py \
    --seed 0 \
    --mode "test" \
    --model-name "SeisMoLLM_emg" \
    --checkpoint  "./checkpoints/model-xx.pth" \
    --log-base "./logs" \
    --log-step 250 \
    --data "../datasets/STEAD/stead_fast_parts" \
    --dataset-name "stead_mag" \
    --data-split true \
    --train-size 0.1 \
    --val-size 0.05 \
    --time-threshold 0.1 \
    --shuffle true \
    --workers 6 \
    --in-samples 6000 \
    --batch-size 128 \
    > ./logs/test_xx_fewshot 2>&1

