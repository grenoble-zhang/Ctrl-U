export MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export CONTROLNET_DIR="work_dirs/reward_model/Captioned_ADE20K/continue-4w8-uncertainty-output_a1r0.1/continue-4w8-uncertainty-output-a1-r0.1_Aug03_17-35-55/checkpoint-32000/controlnet"
export CONTROLNET_DIR="work_dirs/finetune/Captioned_ADE20K/ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16/checkpoint-5000/controlnet"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_ADE20K/time300-uncertainty-output_a1r0.1"

export HF_ENDPOINT=https://hf-mirror.com

# export WANDB_DISABLED=1
export WANDB_API_KEY=77cd7911684a582e22dab946e0fe244df0c3dffa
# export WANDB_ENTITY=c7w
# export CUDA_VISIBLE_DEVICES=6

# export CUDA_LAUNCH_BLOCKING=1

# conda activate /data/Tsinghua/Share/miniconda3/envs/cnpp
# cd /data/Tsinghua/chihh/ControlNet_Plus_Plus 

# Download our fine-tuned weights
# You can also train a new one with command `bash train/finetune_ade20k.sh`
# python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='limingcv/reward_controlnet', filename='diffusion_pytorch_model.safetensors', subfolder=f'${CONTROLNET_DIR}', local_dir='./', local_dir_use_symlinks=False); hf_hub_download(repo_id='limingcv/reward_controlnet', filename='config.json', subfolder=f'${CONTROLNET_DIR}', local_dir='./', local_dir_use_symlinks=False)"

# --dataset_name="/Node10_nvme/Captioned_ADE20K/data" \
# reward fine-tuning
accelerate launch --config_file "train/config2.yml" \
 --main_process_port=$(python3 random_port.py) train/reward_control2.py \
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
 --train_batch_size=6 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=50001 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=2000 \
 --checkpointing_steps=5000 \
 --use_ema \
 --validation_steps=500 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --grad_scale=0.1 \
 --exp_name="xxx" \
 --uncertainty \
 --add_timestep=1 \
 --uncertainty_scale=0.1 \
 --tune_sd="no" $*

#-----------------------------------------------------------------------time---------------------------------------------------------------------------
# #time 300 a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 300 --exp_name="time300-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time300-uncertainty-output_a1r0.1"
# #time 500 a1r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 500 --exp_name="time500-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time500-uncertainty-output_a1r0.1"
# #time 600 a1r0.1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --exp_name="time600-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_a1r0.1"

# 以上三个中途断了，所以拿之前训练的continue下，主要改REWARDMODEL_DIR和lr_warmup_steps置0
# #time 300 a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 300 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="time300-continue3w5-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time300-continue3w5-uncertainty-output_a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time300-uncertainty-output_a1r0.1/time300-uncertainty-output_a1r0.1_Aug17_11-47-22/checkpoint-35000/controlnet"
# #time 500 a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 500 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="time500-continue3w5-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time500-continue3w5-uncertainty-output_a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time500-uncertainty-output_a1r0.1/time500-uncertainty-output_a1r0.1_Aug17_11-47-25/checkpoint-35000/controlnet"
# #time 600 a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="time600-continue3w5-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-continue3w5-uncertainty-output_a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_a1r0.1/time600-uncertainty-output_a1r0.1_Aug17_11-47-28/checkpoint-35000/controlnet"




# #time 700 a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 700 --exp_name="time700-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time700-uncertainty-output_a1r0.1"
# #time 800 a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 800 --exp_name="time800-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time800-uncertainty-output_a1r0.1"
# #time 900 a1r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 900 --exp_name="time900-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time900-uncertainty-output_a1r0.1"
# #time 1000 a1r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 1000 --exp_name="time1000-uncertainty-output_a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time1000-uncertainty-output_a1r0.1"

# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√

#-----------------------------------------------------------------------add_timestep-----------------------------------------------------------------------------
# #time 400 a3r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 3 --exp_name="time400-uncertainty-output_a3r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a3r0.1"
# #time 400 a5r0.1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 5 --exp_name="time400-uncertainty-output_a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a5r0.1"
# #time 400 a5r0.1 continue3w
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 5 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="ade20k-time400-continue3w-uncertainty-output_a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w-uncertainty-output_a5r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a5r0.1/time400-uncertainty-output_a5r0.1_Aug19_13-20-15/checkpoint-30000/controlnet"




# time 400 a5r0.1 1w continue
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 5 --lr_warmup_steps 0 --max_train_steps 40001 --exp_name="time400-continue1w-uncertainty-output_a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue1w-uncertainty-output_a5r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a5r0.1/time400-uncertainty-output_a5r0.1_Aug19_13-20-15/checkpoint-10000/controlnet"
# time 400 a5r0.1 2w5 continue
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 5 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="time400-continue2w5-uncertainty-output_a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue2w5-uncertainty-output_a5r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-continue1w-uncertainty-output_a5r0.1/time400-continue1w-uncertainty-output_a5r0.1_Aug20_13-45-43/checkpoint-15000/controlnet"
# time 400 a5r0.1 3w continue
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 5 --lr_warmup_steps 0 --max_train_steps 15001 --exp_name="time400-continue3w-uncertainty-output_a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w-uncertainty-output_a5r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a5r0.1/time400-uncertainty-output_a5r0.1_Aug19_13-20-15/checkpoint-30000/controlnet"
# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√

# #time 400 a7r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 7 --exp_name="time400-uncertainty-output_a7r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a7r0.1"
# #time 400 a7r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 7 --lr_warmup_steps 0 --max_train_steps 10001 --exp_name="ade20k-time400-continue3w5-uncertainty-output_a7r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w5-uncertainty-output_a7r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a7r0.1/time400-uncertainty-output_a7r0.1_Aug20_21-19-07/checkpoint-35000/controlnet"

# #time 400 a9r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 9 --exp_name="time400-uncertainty-output_a9r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a9r0.1"
# #time 400 a9r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --add_timestep 9 --lr_warmup_steps 0 --max_train_steps 10001 --exp_name="ade20k-time400-continue3w5-uncertainty-output_a9r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w5-uncertainty-output_a9r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_a9r0.1/time400-uncertainty-output_a9r0.1_Aug20_21-19-29/checkpoint-35000/controlnet"



# time400 g0.1a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --exp_name="time400-uncertainty-output_g0.1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a1r0.1"
# time400 g0.1a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --lr_warmup_steps 0 --max_train_steps 10001 --exp_name="ade20k-time400-uncertainty-continue3w5-output_g0.1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w5-uncertainty-output_g0.1a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a1r0.1/time400-uncertainty-output_g0.1a1r0.1_Aug20_21-19-33/checkpoint-35000/controlnet"
# time400 g0.1a5r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 5 --exp_name="time400-uncertainty-output_g0.1a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a5r0.1"

# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√

# time400 g0.05a1r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.05 --add_timestep 1 --exp_name="time400-uncertainty-output_g0.05a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.05a1r0.1"
# time400 g0.05a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.05 --add_timestep 1 --lr_warmup_steps 0 --max_train_steps 10001 --exp_name="ade20k-time400-uncertainty-continue3w5-output_g0.05a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w5-uncertainty-output_g0.05a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.05a1r0.1/time400-uncertainty-output_g0.05a1r0.1_Aug21_00-31-56/checkpoint-35000/controlnet"


# time400 g1a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 1 --add_timestep 1 --exp_name="time400-uncertainty-output_g1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g1a1r0.1"
# time400 g1a1r0.1 continue3w5
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 1 --add_timestep 1 --lr_warmup_steps 0 --max_train_steps 10001 --exp_name="ade20k-time400-uncertainty-continue3w5-output_g1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-continue3w5-uncertainty-output_g1a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g1a1r0.1/time400-uncertainty-output_g1a1r0.1_Aug21_00-32-01/checkpoint-35000/controlnet"



# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√


# time200 g0.1a1r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --exp_name="ade20k-time200-uncertainty-output_g0.1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time200-uncertainty-output_g0.1a1r0.1"
# time200 g0.1a1r0.1 continue1w
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 200 --grad_scale 0.1 --add_timestep 1 --lr_warmup_steps 0 --max_train_steps 35001 --exp_name="ade20k-time200-uncertainty-continue1w-output_g0.1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time200-continue1w-uncertainty-output_g0.1a1r0.1" --controlnet_model_name_or_path="/Node11_nvme/zhanggy/code/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/time200-uncertainty-output_g0.1a1r0.1/ade20k-time200-uncertainty-output_g0.1a1r0.1_Aug22_22-59-09/checkpoint-10000/controlnet"

# time200 g0.05a1r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 200 --grad_scale 0.05 --add_timestep 1 --exp_name="ade20k-time200-uncertainty-output_g0.05a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time200-uncertainty-output_g0.05a1r0.1"

# time200 g1a1r0.1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 200 --grad_scale 1 --add_timestep 1 --exp_name="ade20k-time200-uncertainty-output_g1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time200-uncertainty-output_g1a1r0.1"
# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√

# time600 g0.05a1r0.1
# CUDA_VISIBLE_DEVICES=4,5 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --grad_scale 0.05 --add_timestep 1 --exp_name="ade20k-time600-uncertainty-output_g0.05a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_g0.05a1r0.1"
# time600 g0.1a1r0.1
# CUDA_VISIBLE_DEVICES=6,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --grad_scale 0.1 --add_timestep 1 --exp_name="ade20k-time600-uncertainty-output_g0.1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_g0.1a1r0.1"
# time600 g1a1r0.1
# CUDA_VISIBLE_DEVICES=2,3 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --grad_scale 1 --add_timestep 1 --exp_name="ade20k-time600-uncertainty-output_g1a1r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_g1a1r0.1"
# time600 g0.1a5r0.1
# CUDA_VISIBLE_DEVICES=0,1 bash train/reward_ade20k_out.sh --max_timestep_rewarding 600 --grad_scale 0.1 --add_timestep 5 --exp_name="ade20k-time600-uncertainty-output_g0.1a5r0.1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time600-uncertainty-output_g0.1a5r0.1"
# √√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√



# time400 g0.1a1r0.5
# CUDA_VISIBLE_DEVICES=3,4 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 0.5 --exp_name="ade20k-time400-uncertainty-output_g0.1a1r0.5" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a1r0.5"
# time400 g0.1a1r1
# CUDA_VISIBLE_DEVICES=5,6 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 1 --exp_name="ade20k-time400-uncertainty-output_g0.1a1r1" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a1r1"

# time400 g0.1a1r0.05
# CUDA_VISIBLE_DEVICES=5,7 bash train/reward_ade20k_out.sh --max_timestep_rewarding 400 --grad_scale 0.1 --add_timestep 1 --uncertainty_scale 0.05 --exp_name="ade20k-time400-uncertainty-output_g0.1a1r0.05" --output_dir="work_dirs/reward_model/Captioned_ADE20K/time400-uncertainty-output_g0.1a1r0.05"