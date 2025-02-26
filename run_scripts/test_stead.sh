export OMP_NUM_THREADS=6

dt=`date +'%Y-%m-%d_%H-%M-%S'`

python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "SeisGPT_dpk" \
    --checkpoint  "./checkpoints/model-xx.pth" \
    --log-base "./logs" \
    --log-step 250 \
    --data "../datasets/STEAD" \
    --dataset-name "stead" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --time-threshold 0.1 \
    --shuffle true \
    --workers 6 \
    --in-samples 6000 \
    --batch-size 128 \
    > ./logs/test_xx 2>&1

