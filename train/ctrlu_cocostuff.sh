export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="checkpoint/finetune/cocostuff"
export REWARDMODEL_DIR="mmseg::deeplabv3/deeplabv3_r50-d8_4xb4-160k_coco-stuff164k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_COCOStuff/ctrl-u"

accelerate launch --config_file "train/config.yml" \
 --main_process_port=$(python3 random_port.py) train/ctrlu-reward.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="limingcv/Captioned_COCOStuff" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --label_column="panoptic_seg_map" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=40001 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=2000 \
 --checkpointing_steps=5000 \
 --use_ema \
 --validation_steps=2000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --grad_scale=0.2 \
 --add_timestep=1 \
 --u_scale=0.1 \