DEVICE="cuda"
BATCH_SIZE="32"
ENTRY='main.py'
MAX_ITER=30

# for TASK in task2_merged
for TASK in task1 task2_merged
do
    # train image-only model (densenet201)
    # python $ENTRY --model_name "image_only_${TASK}" --mode image_only --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER

    # train text-only model (bert)
    # python $ENTRY --model_name "text_only_${TASK}" --mode text_only --task $TASK --batch_size 32 --device $DEVICE --max_iter $MAX_ITER

    # Combine them together
    # python $ENTRY --model_name "full_${TASK}" --mode both --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
    # --image_model_to_load "./output/image_only_${TASK}/best.pt"  --text_model_to_load "./output/text_only_${TASK}/best.pt" --consistent_only

    # Use category info
    # python $ENTRY --model_name "full_${TASK}_cate" --mode both --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
    # --image_model_to_load "./output/image_only_${TASK}/best.pt"  --text_model_to_load "./output/text_only_${TASK}/best.pt" --use_cate --consistent_only

    # Use selected (improved) category info
    # python $ENTRY --model_name "full_${TASK}_cate_selected" --mode both --task $TASK --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
    # --image_model_to_load "./output/image_only_${TASK}/best.pt"  --text_model_to_load "./output/text_only_${TASK}/best.pt" --use_cate --consistent_only
done

# Use selected (improved) category info
python $ENTRY --model_name "full_task1_cate_selected" --mode both --task task1 --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--image_model_to_load "./output/image_only_task1/best.pt"  --text_model_to_load "./output/text_only_task1/best.pt" --use_cate --consistent_only --cate_dim 21

python $ENTRY --model_name "full_task2_merged_cate_selected" --mode both --task task2_merged --batch_size 16 --device $DEVICE --max_iter $MAX_ITER \
--image_model_to_load "./output/image_only_task2_merged/best.pt"  --text_model_to_load "./output/text_only_task2_merged/best.pt" --use_cate --consistent_only --cate_dim 20

