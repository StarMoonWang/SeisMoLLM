export OMP_NUM_THREADS=6

dt=`date +'%Y-%m-%d_%H-%M-%S'`

for tt in 0.1
do

torchrun --nnodes 1 --nproc_per_node 4 main.py \
    --seed 0 \
    --mode "test" \
    --model-name "SeisMoLLM_dpk" \
    --checkpoint  "./checkpoints/model-xx.pth" \
    --log-base "./logs" \
    --log-step 700 \
    --data "/datasets/DiTing330km" \
    --dataset-name "diting_fewshot" \
    --data-split true \
    --train-size 0.0 \
    --val-size 0.0 \
    --time-threshold $tt \
    --shuffle true \
    --workers 6 \
    --batch-size 256 \
    > ./logs/test_xx_fewshot 2>&1

done 
