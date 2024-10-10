export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="checkpoints/canny/controlnet"
export REWARDMODEL_DIR="canny"
export OUTPUT_DIR="work_dirs/reward_model/Canny_after_ft/after_ft_time200_uncertainty_output_g0.1a1r1"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"

accelerate launch --config_file "train/config2.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control-canny.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="canny" \
 --dataset_name="/Node10_nvme/MultiGen-20M_train/data" \
 --caption_column="text" \
 --conditioning_image_column="canny" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=6 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=80001 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=0 \
 --checkpointing_steps=2000 \
 --use_ema \
 --validation_steps=2000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --uncertainty \
 --grad_scale=1.0 \
 --add_timestep=1 \
 --uncertainty_scale=0.1 \
 --exp_name="xxx" \
 --tune_sd="no" $*


#debug 
# CUDA_VISIBLE_DEVICES=4 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="canny_after_ft_uncertainty_output_time200g1.0a1r0.1" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g1.0a1r0.1"

# time200 g1a1r0.1 ×
# CUDA_VISIBLE_DEVICES=0,2 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="canny_after_ft_uncertainty_output_time200g1a1r0.1" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g1a1r0.1"

# time200 g1a1r0.01 ×
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.01 --exp_name="canny_after_ft_uncertainty_output_time200g1a1r0.01" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g1a1r0.01"

# time200 g1a1r0.001 ×
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.001 --exp_name="canny_after_ft_uncertainty_output_time200g1a1r0.001" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g1a1r0.001"


# time200 g0.1a1r0.1 ×
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="canny_after_ft_uncertainty_output_time200g0.1a1r0.1" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g0.1a1r0.1"

# time200 g0.1a1r0.01 ×
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 0.01 --exp_name="canny_after_ft_uncertainty_output_time200g0.1a1r0.01" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g0.1a1r0.01"






# time200 g1a1r0.1 ×
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 1.0 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="canny_after_ft_uncertainty_output_time200g1.0a1r0.1" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g1.0a1r0.1"





# time200 g0.1a1r1 ×
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="canny_after_ft_uncertainty_output_time200g0.1a1r1" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g0.1a1r1"





# time100 g1a1r0.0001
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 100 --grad_scale 1 --add_timestep 1 --uncertainty_scale 0.0001 --exp_name="canny_after_ft_uncertainty_output_time100g1a1r0.0001" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time100g1a1r0.0001"

# time100 g1a1r0.00001
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 100 --grad_scale 1 --add_timestep 1 --uncertainty_scale 0.00001 --exp_name="canny_after_ft_uncertainty_output_time100g1a1r0.00001" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time100g1a1r0.00001"


# time200 g0.5a1r0.0001
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.5 --add_timestep 1 --uncertainty_scale 0.0001 --exp_name="canny_after_ft_uncertainty_output_time200g0.5a1r0.0001" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g0.5a1r0.0001"

# time200 g0.5a1r0.00001
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_canny_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.5 --add_timestep 1 --uncertainty_scale 0.00001 --exp_name="canny_after_ft_uncertainty_output_time200g0.5a1r0.00001" --output_dir="work_dirs/reward_model/Canny_after_ft/after_ft_uncertainty_output_time200g0.5a1r0.00001"