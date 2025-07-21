export OMP_NUM_THREADS=6

dt=`date +'%m-%d_%H-%M-%S'`

# tar -xf ../datasets/STEAD/stead.tar -C /dev/shm

torchrun --nnodes 1 --nproc_per_node 4 --master_port 10001 main.py \
    --seed 0 \
    --mode "train" \
    --model-name "SeisMoLLM_dis" \
    --log-base "./logs" \
    --log-step 500 \
    --data "/dev/shm/stead" \
    --dataset-name "stead" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle true \
    --workers 6 \
    --in-samples 6000 \
    --batch-size 256 \
    --augmentation true \
    --epochs 200 \
    --patience 30 \
    --base-lr 0.0005 \
    --max-lr 0.001 \
    --warmup-steps 4500 \
    --down-steps 5000 \
    > ./train_stead_dis 2>&1

    # --start-epoch x \
    # --checkpoint "./x/model-x.pth" \
