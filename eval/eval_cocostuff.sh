CONTROLNET_DIR=$1  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=1
# Guidance scale and inference steps
export SCALE=$3
export NUM_STEPS=20
export HF_ENDPOINT=https://hf-mirror.com

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --config_file "train/debug.yml" --main_process_port=$(python3 random_port.py) --num_processes=$NUM_GPUS eval/eval.py --task_name='seg' --dataset_name='/Node10_nvme/Captioned_COCOStuff/data' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='panoptic_seg_map' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name $2

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/Captioned_COCOStuff/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Evaluation with mmseg api
# mim test mmseg mmlab/mmseg/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py \
#     --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth \
#     --gpus 4 \
#     --launcher pytorch \
#     --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
#                   work_dir="${DATA_DIR}"