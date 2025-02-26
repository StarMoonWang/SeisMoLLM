export OMP_NUM_THREADS=6

dt=`date +'%m-%d_%H-%M-%S'`

torchrun --nnodes 1 --nproc_per_node 4 --master_port 10002 main.py \
    --seed 0 \
    --mode "train" \
    --model-name "SeisGPT_baz" \
    --log-base "./logs" \
    --log-step 500 \
    --data "/datasets/stead" \
    --dataset-name "stead" \
    --data-split true \
    --train-size 0.1 \
    --val-size 0.05 \
    --shuffle true \
    --workers 6 \
    --in-samples 6000 \
    --batch-size 128 \
    --augmentation true \
    --epochs 200 \
    --patience 30 \
    --base-lr 0.0005 \
    --max-lr 0.001 \
    --warmup-steps 600 \
    --down-steps 800 \
    > ./train_stead_baz_fewshot 2>&1

    # --start-epoch x \
    # --checkpoint "./x/model-x.pth" \
