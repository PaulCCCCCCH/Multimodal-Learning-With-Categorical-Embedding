DEVICE="cuda"
BATCH_SIZE="32"
ENTRY='main.py'
MAX_ITER=40

# Use selected category info
python $ENTRY --model_name "full_task1_cate_selected" --mode both --task task1 --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--model_to_load "./output/full_task1_cate_selected/best.pt" --no_transform --use_cate --eval --consistent_only --cate_dim 21
# Use selected category info
python $ENTRY --model_name "full_task2_merged_cate_selected" --mode both --task task2_merged --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--model_to_load "./output/full_task2_merged_cate_selected/best.pt" --no_transform --use_cate --eval --consistent_only --cate_dim 20
