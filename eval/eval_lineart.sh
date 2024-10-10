# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"  # Eval ControlNet
CONTROLNET_DIR=$1  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=6
# Guidance scale and inference steps
export SCALE=$3
export NUM_STEPS=20
export HF_ENDPOINT=https://hf-mirror.com

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --config_file "train/config6.yml" --main_process_port=$(python3 random_port.py) --num_processes=$NUM_GPUS eval/eval.py --task_name='lineart' --dataset_name='/Node10_nvme/MultiGen-20M_canny_eval/data' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name $2

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_canny_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Run the evaluation code
# python3 eval/eval_edge.py --task lineart --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}