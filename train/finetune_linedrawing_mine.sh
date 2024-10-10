export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_lineart"
export REWARDMODEL_DIR="https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth"
export REWARDMODEL_DIR="https://hf-mirror.com/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth"
export OUTPUT_DIR="work_dirs/finetune/MultiGen20M_LineDrawing/no/"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
export HF_DATASETS_CACHE="/Node09_nvme/zhanggy/CPP_Uncertainty/.cache"

accelerate launch --config_file "train/config4.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="lineart" \
 --dataset_name="/Node10_nvme/MultiGen-20M_train/data" \
 --caption_column="text" \
 --conditioning_image_column="lineart" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=32 \
 --max_train_steps=100000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=1000 \
 --checkpointing_steps=5000 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --use_ema \
 --grad_scale=0.0 \
 --exp_name="test" \
 --tune_sd="no" $*

# bash train/finetune_linedrawing.sh --exp_name="shift1.0_no" --pretrained_unet_path="work_dirs/fm_sd21"


# #  --pretrained_model_name_or_path=$MODEL_DIR \
#  --controlnet_model_name_or_path=$CONTROLNET_DIR \

 # 1.5
 # no with control model
 # CUDA_VISIBLE_DEVICES=4,5,6,7 bash train/finetune_linedrawing_mine.sh --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_lineart" --output_dir="work_dirs/finetune/MultiGen20M_linedrawing/1-5/no-with" --tune_sd="no" --exp_name="MultiGen20M_linedrawing-1.5-no-with"
 # no without control model
 # CUDA_VISIBLE_DEVICES=4,5,6,7 bash train/finetune_linedrawing_mine.sh --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="work_dirs/finetune/MultiGen20M_linedrawing/1-5/no-without" --tune_sd="no" --exp_name="MultiGen20M_linedrawing-1.5-no-without"

# 2.1
 # no with control model
 # CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/finetune_linedrawing_mine.sh --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" --controlnet_model_name_or_path="work_dirs/finetune/MultiGen20M_linedrawing/2-1/with-controlnet" --output_dir="work_dirs/finetune/MultiGen20M_linedrawing/2-1/no-with" --tune_sd="no" --exp_name="MultiGen20M_linedrawing-2.1-no-with"
 # no without control model
 # CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/finetune_linedrawing_mine.sh --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" --output_dir="work_dirs/finetune/MultiGen20M_linedrawing/2-1/no-without" --tune_sd="no" --exp_name="MultiGen20M_linedrawing-2.1-no-without"