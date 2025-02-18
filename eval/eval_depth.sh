export CONTROLNET_DIR="checkpoint/reward/depth"
export SCALE=3.0
export NUM_STEPS=20
export exp_name="depth-eval"
export HF_ENDPOINT="https://hf-mirror.com"
# Generate images for evaluation
accelerate launch --config_file "config.yml" --main_process_port=$(python3 random_port.py) --num_processes=1 eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text' --label_column='control_depth' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name=${exp_name}

export DATA_DIR="work_dirs/eval_dirs/data/validation/${exp_name}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}