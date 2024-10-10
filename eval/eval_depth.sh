# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"  # Eval ControlNet
CONTROLNET_DIR=$1  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=8
# Guidance scale and inference steps
export SCALE=$3
export NUM_STEPS=20
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch --config_file "train/config8.yml" --main_process_port=$(python3 random_port.py) --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name="/Node10_nvme/MultiGen-20M_depth_eval/data" --dataset_split='validation' --condition_column='control_depth' --prompt_column='text' --label_column='control_depth' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name $2
