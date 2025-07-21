export OMP_NUM_THREADS=6

dt=`date +'%m-%d_%H-%M-%S'`

torchrun --nnodes 1 --nproc_per_node 4 --master_port 10000 main.py \
    --seed 0 \
    --mode "train" \
    --model-name "SeisMoLLM_baz" \
    --log-base "./logs" \
    --log-step 300 \
    --data "/datasets/DiTing330km" \
    --dataset-name "diting_light" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle true \
    --workers 6 \
    --in-samples 8192 \
    --batch-size 128 \
    --augmentation true \
    --epochs 200 \
    --patience 30 \
    --base-lr 0.0005 \
    --max-lr 0.001 \
    --warmup-steps 2500 \
    --down-steps 3000 \
    > ./train_diting_baz 2>&1

    # --start-epoch x \
    # --checkpoint "./x/model-x.pth" \
