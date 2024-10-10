export MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export UNET_DIR="work_dirs/finetune/Captioned_COCOStuff/1-5/no-with/Captioned_COCOStuff-1.5-no-with_Aug28_14-51-08/checkpoint-35001/UNet2DConditionModel"
export CONTROLNET_DIR="work_dirs/finetune/Captioned_COCOStuff/1-5/no-with/Captioned_COCOStuff-1.5-no-with_Aug28_14-51-08/checkpoint-35001/ControlNetModel"
export REWARDMODEL_DIR="mmseg::deeplabv3/deeplabv3_r50-d8_4xb4-160k_coco-stuff164k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_COCOStuff/after_ft_time400_uncertainty_output_g0.1a1r1"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"

accelerate launch --config_file "train/config4.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_unet_path=$UNET_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="/Node10_nvme/Captioned_COCOStuff/data" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --label_column="panoptic_seg_map" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=6 \
 --gradient_accumulation_steps=2 \
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
 --grad_scale=0.2 \
 --add_timestep=1 \
 --uncertainty_scale=0.1 \
 --exp_name="xxx" \
 --tune_sd="no" $*


# time200 g0.2a1r0.1
# CUDA_VISIBLE_DEVICES=0,2 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.2 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="coco_after_ft_uncertainty_output_time200g0.2a1r0.1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time200g0.2a1r0.1"

# CUDA_VISIBLE_DEVICES=0,2 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 800 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="coco_after_ft_uncertainty_output_time800g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time800g0.1a1r1"


# time200 g0.2a0r0.1
# CUDA_VISIBLE_DEVICES=1,2,5,6 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.2 --add_timestep 0 --uncertainty_scale 0.1 --exp_name="coco_after_ft_uncertainty_output_time200g0.2a0r0.1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time200g0.2a0r0.1"







# time200 g0.1a1r1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="coco_after_ft_uncertainty_output_time200g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time200g0.1a1r1"




# time500 g0.1a1r1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 500 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="coco_after_ft_uncertainty_output_time500g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time500g0.1a1r1"

# time600 g0.1a1r1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 600 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="coco_after_ft_uncertainty_output_time600g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time600g0.1a1r1"

# time500 g0.2a1r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 500 --grad_scale 0.2 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="coco_after_ft_uncertainty_output_time500g0.2a1r0.1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time500g0.2a1r0.1"
# time600 g0.2a1r0.1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_cocostuff_mine_1_5.sh --max_timestep_rewarding 600 --grad_scale 0.2 --add_timestep 1 --uncertainty_scale 0.1 --exp_name="coco_after_ft_uncertainty_output_time600g0.2a1r0.1" --output_dir="work_dirs/reward_model/Captioned_COCOStuff_after_ft/after_ft_uncertainty_output_time600g0.2a1r0.1"