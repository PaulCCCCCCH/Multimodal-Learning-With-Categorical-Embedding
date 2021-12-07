DEVICE="cuda"
BATCH_SIZE="32"
ENTRY='main.py'
MAX_ITER=30

# Use selected (improved) category info
python $ENTRY --model_name "full_task1_cate_selected" --mode both --task task1 --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--image_model_to_load "./output/image_only_task1/best.pt"  --text_model_to_load "./output/text_only_task1/best.pt" --use_cate --consistent_only --cate_dim 21

python $ENTRY --model_name "full_task2_merged_cate_selected" --mode both --task task2_merged --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--image_model_to_load "./output/image_only_task2_merged/best.pt"  --text_model_to_load "./output/text_only_task2_merged/best.pt" --use_cate --consistent_only --cate_dim 20

