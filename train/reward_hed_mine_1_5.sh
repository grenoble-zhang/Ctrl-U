export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export UNET_DIR="work_dirs/finetune/MultiGen20M_hed/1-5/no-with/MultiGen20M_hed-1.5-no-with_Sep04_11-22-56/checkpoint-50001/UNet2DConditionModel"
export CONTROLNET_DIR="work_dirs/finetune/MultiGen20M_hed/1-5/no-with/MultiGen20M_hed-1.5-no-with_Sep04_11-22-56/checkpoint-50001/ControlNetModel"
export REWARDMODEL_DIR="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
export OUTPUT_DIR="work_dirs/reward_model/Hed_after_ft/after_ft_time200_uncertainty_output_g0.1a1r1"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"



accelerate launch --config_file "train/config4.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control-hed.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_unet_path=$UNET_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="hed" \
 --dataset_name="/Node10_nvme/MultiGen-20M_train/data" \
 --caption_column="text" \
 --conditioning_image_column="hed" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=3 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=60001 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=2000 \
 --checkpointing_steps=5000 \
 --use_ema \
 --validation_steps=2000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=400 \
 --uncertainty \
 --grad_scale=1.0 \
 --add_timestep=1 \
 --uncertainty_scale=0.1 \
 --exp_name="xxx" \
 --tune_sd="no" $*

# time400 g1a1r0.1
# CUDA_VISIBLE_DEVICES=2,5,6 bash train/reward_hed_mine_1_5.sh --max_timestep_rewarding 400 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="hed_after_ft_uncertainty_output_time400g1a1r0.1" --output_dir="work_dirs/reward_model/Hed_after_ft/after_ft_uncertainty_output_time400g1a1r0.1"

# time400 g1a0r0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/reward_hed_mine_1_5.sh --max_timestep_rewarding 400 --grad_scale 1.0 --add_timestep 0 --uncertainty_scale 0.1 --exp_name="hed_after_ft_uncertainty_output_time400g1a0r0.1" --output_dir="work_dirs/reward_model/Hed_after_ft/after_ft_uncertainty_output_time400g1a0r0.1"








# time400 g0.1a1r0.1
# CUDA_VISIBLE_DEVICES=4,5,6 bash train/reward_hed_mine_1_5.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="hed_after_ft_uncertainty_output_time400g0.1a1r0.1" --output_dir="work_dirs/reward_model/Hed_after_ft/after_ft_uncertainty_output_time400g0.1a1r0.1"