export CONTROLNET_DIR="checkpoint/reward/ade20k"
export SCALE=2.0
export NUM_STEPS=20
export exp_name="ade20k-eval"
export HF_ENDPOINT="https://hf-mirror.com"
# Generate images for evaluation
accelerate launch --config_file "config.yml" --main_process_port=$(python3 random_port.py) --num_processes=1 eval/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_ADE20K' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='seg_map' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name=${exp_name}

export DATA_DIR="work_dirs/eval_dirs/data/validation/${exp_name}"

# Evaluation with mmseg api
mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
    --gpus 1 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
                  work_dir="${DATA_DIR}"