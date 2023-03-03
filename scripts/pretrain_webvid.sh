# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/checkpoints/log'
# Set the path to Kinetics train set. 
DATA_PATH='YOUR_PATH/dataset/webvid/train.json'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)

OMP_NUM_THREADS=1 /apdcephfs/private_mayuema/envs/vmae/bin/python -u -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12322 --nnodes=8 \
        --node_rank=$1 --master_addr=$2 \
        run_simvtp_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_simvtp_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 1 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}\
        --lr 1.5e-4 >> $(pwd)"/log.log"



        