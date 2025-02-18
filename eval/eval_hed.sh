export CONTROLNET_DIR="checkpoint/reward/hed"
export SCALE=7.5
export NUM_GPUS=1
export NUM_STEPS=20
export exp_name="hed-eval"
export HF_ENDPOINT="https://hf-mirror.com"
# Generate images for evaluation
accelerate launch --config_file "config.yml" --main_process_port=$(python3 random_port.py) --num_processes=1 eval/eval.py --task_name='hed' --dataset_name="limingcv/MultiGen-20M_canny_eval" --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name=${exp_name}

export DATA_DIR="work_dirs/eval_dirs/data/validation/${exp_name}"

# Run the evaluation code
python3 eval/eval_edge.py --task hed --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}