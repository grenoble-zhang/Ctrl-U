export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export UNET_DIR="/Node10_nvme/zhanggy/CPP_Uncertainty/work_dirs/finetune/Captioned_ADE20K/1-5/no-with/Captioned_ADE20K-1.5-no-with_Aug28_03-13-25/checkpoint-15001/UNet2DConditionModel"
export CONTROLNET_DIR="/Node10_nvme/zhanggy/CPP_Uncertainty/work_dirs/finetune/Captioned_ADE20K/1-5/no-with/Captioned_ADE20K-1.5-no-with_Aug28_03-13-25/checkpoint-15001/ControlNetModel"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_ADE20K/after_ft_time400_uncertainty_output_g0.1a1r1"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"

accelerate launch --config_file "train/config8.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_unet_path=$UNET_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="/Node10_nvme/Captioned_ADE20K/data" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=6 \
 --gradient_accumulation_steps=1 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=50001 \
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
 --grad_scale=0.1 \
 --add_timestep=1 \
 --uncertainty_scale=1 \
 --exp_name="xxx" \
 --tune_sd="no" $*


# time400 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=1,3 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 400 --exp_name="ade20k_after_ft_time400_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time400_uncertainty_output_g0.1a1r1"

# time400 g0.1a1r1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 400 --exp_name="ade20k_after_ft_time400_uncertainty_output_g0.1a0r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time400_uncertainty_output_g0.1a0r1"

# time500 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 500 --exp_name="ade20k_after_ft_time500_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time500_uncertainty_output_g0.1a1r1"


# time600 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 600 --exp_name="ade20k_after_ft_time600_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time600_uncertainty_output_g0.1a1r1"



# time700 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 700 --exp_name="ade20k_after_ft_time700_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time700_uncertainty_output_g0.1a1r1"


# time800 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 800 --exp_name="ade20k_after_ft_time800_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time800_uncertainty_output_g0.1a1r1"



# time900 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=1,3 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 900 --exp_name="ade20k_after_ft_time900_uncertainty_output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/after_ft_time900_uncertainty_output_g0.1a1r1"




# time900 g0.1a1r1 √
# CUDA_VISIBLE_DEVICES=6 bash train/reward_ade20k_mine_1_5.sh --max_timestep_rewarding 900 --exp_name="debug" --output_dir="work_dirs/reward_model/Captioned_ADE20K/debug"
