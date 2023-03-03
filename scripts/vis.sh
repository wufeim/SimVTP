# Set the path to save video
OUTPUT_DIR='YOUR_PATH/SimVTP/visualization'
# path to video for visualization
VIDEO_PATH='YOUR_PATH/dataset/VIDEO_PATH'
# path to pretrain model
MODEL_PATH='YOUR_PATH/checkpoints'

python run_simvtp_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_simvtp_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}