# Set the path to save checkpoints
OUTPUT_DIR='/apdcephfs/private_mayuema/mayuema/checkpoints/debug'
# Set the path to Kinetics train set. 
DATA_PATH='/apdcephfs/private_mayuema/dataset/MSRVTT/train_9k_miech.json'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)

OMP_NUM_THREADS=1 /apdcephfs/private_mayuema/envs/vmae/bin/python -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12321 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        run_simvtp_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_simvtp_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 2 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}



        