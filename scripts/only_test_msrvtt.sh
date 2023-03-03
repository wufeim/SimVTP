# Set the path to save checkpoints
OUTPUT_DIR='YOUR_SIMVTP_PATH/SimVTP/checkpoints/debug_test'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_SIMVTP_PATH/dataset/MSRVTT'
# path to pretrain model
# MODEL_PATH='/apdcephfs/private_mayuema/v1_videomae_pretrain/checkpoints/version_0_pretrain_100e/checkpoint-99.pth'
MODEL_PATH='YOUR_SIMVTP_PATH/WEIGHT'

# batch_size can be adjusted according to number of GPUs
OMP_NUM_THREADS=1  /apdcephfs/private_mayuema/envs/vmae/bin/python  -u -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 23461 --nnodes=1  --node_rank=$1 --master_addr=$2  \
    run_simvtp_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set MSRVTT \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --only_test \
    --test_num_segment 15 \
    --test_num_crop 3 \
    --enable_deepspeed  