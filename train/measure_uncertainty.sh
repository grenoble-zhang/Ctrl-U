export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="/Node10_nvme/zhanggy/CPP_Uncertainty/work_dirs/finetune/Captioned_ADE20K/ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16/checkpoint-5000/controlnet"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/measure_uncertainty"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"

accelerate launch --config_file "train/debug.yml" \
 --main_process_port=$(python3 random_port.py) train/measure_uncertainty.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="/Node10_nvme/Captioned_ADE20K/data" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1000 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --lr_scheduler="constant_with_warmup" \
 --num_train_epochs=1 \
 --lr_warmup_steps=0 \
 --checkpointing_steps=1000000 \
 --use_ema \
 --validation_steps=1000000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=1000 \
 --uncertainty \
 --grad_scale=0.1 \
 --step_number=50 \
 --add_timestep=10 \
 --uncertainty_scale=1 \
 --exp_name="xxx" $*

# debug √
# CUDA_VISIBLE_DEVICES=0 bash train/measure_uncertainty.sh --step_number 200 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"
# CUDA_VISIBLE_DEVICES=1 bash train/measure_uncertainty.sh --step_number 300 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"
# CUDA_VISIBLE_DEVICES=2 bash train/measure_uncertainty.sh --step_number 430 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"
# CUDA_VISIBLE_DEVICES=4 bash train/measure_uncertainty.sh --step_number 450 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"






# CUDA_VISIBLE_DEVICES=5 bash train/measure_uncertainty.sh --step_number 550 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"
# CUDA_VISIBLE_DEVICES=6 bash train/measure_uncertainty.sh --step_number 230 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"
# CUDA_VISIBLE_DEVICES=7 bash train/measure_uncertainty.sh --step_number 750 --exp_name="measure_uncertainty" --output_dir="work_dirs/measure_uncertainty/Captioned_ADE20K"