export OMP_NUM_THREADS=6

dt=`date +'%Y-%m-%d_%H-%M-%S'`

for tt in 0.1
do

python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "SeisMoLLM_dpk" \
    --checkpoint  "./checkpoints/model-xx.pth" \
    --log-base "./logs" \
    --log-step 700 \
    --data "/datasets/DiTing330km" \
    --dataset-name "diting_light" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --time-threshold $tt \
    --shuffle true \
    --workers 6 \
    --batch-size 256 \
    > ./logs/test_xx 2>&1

done 
