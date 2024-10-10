# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"  # Eval ControlNet
CONTROLNET_DIR=$1  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=2
# Guidance scale and inference steps
export SCALE=$3
export NUM_STEPS=20
export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME="/Node10_nvme/CPP_Uncertainty/cache"


# bash eval/eval_ade20k.sh --model_path /data/Tsinghua/chihh/CPP_Uncertainty/work_dirs/reward_model/Captioned_ADE20K/1reward_ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_UperNet-R50/uncertainty-1_Jul26_23-11-53/checkpoint-2000/controlnet --exp_name r0

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --config_file "train/config2.yml" --main_process_port=$(python3 random_port.py) --num_processes=$NUM_GPUS eval/eval.py --task_name='seg' --dataset_name="/Node10_nvme/Captioned_ADE20K/data" --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='seg_map' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name $2

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/Captioned_ADE20K/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Evaluation with mmseg api
# mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
#     --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
#     --gpus 8 \
#     --launcher pytorch \
#     --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
#                   work_dir="${DATA_DIR}"