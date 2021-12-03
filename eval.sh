DEVICE="cuda"
BATCH_SIZE="32"
ENTRY='main.py'
MAX_ITER=40

for TASK in task1 task2_merged
do
    # Combine them together
    python $ENTRY --model_name "full_${TASK}" --mode both --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
    --model_to_load "./output/full_${TASK}/best.pt" --no_transform --eval

    # Use category info
    python $ENTRY --model_name "full_${TASK}_cate" --mode both --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
    --model_to_load "./output/full_${TASK}_cate/best.pt" --no_transform --use_cate --eval
done
