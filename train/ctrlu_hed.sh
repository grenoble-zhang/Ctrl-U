export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="checkpoint/finetune/hed"
export REWARDMODEL_DIR="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
export OUTPUT_DIR="work_dirs/reward_model/hed/ctrl-u"

accelerate launch --config_file "train/config.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control-hed.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="hed" \
 --dataset_name="limingcv/MultiGen-20M_train" \
 --caption_column="text" \
 --conditioning_image_column="hed" \
 --resolution=512 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=8 \
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
 --grad_scale=1 \
 --add_timestep=1 \
 --u_scale=0.1 \