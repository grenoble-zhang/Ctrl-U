export CONTROLNET_DIR="checkpoint/reward/cocostuff"
export SCALE=2.0
export NUM_STEPS=20
export exp_name="coco_stuff-eval"
export HF_ENDPOINT="https://hf-mirror.com"
# Generate images for evaluation
accelerate launch --config_file "config.yml" --main_process_port=$(python3 random_port.py) --num_processes=1 eval/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_COCOStuff' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='panoptic_seg_map' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --exp_name=${exp_name}

export DATA_DIR="work_dirs/eval_dirs/data/validation/${exp_name}"

# Evaluation with mmseg api
mim test mmseg mmlab/mmseg/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth \
    --gpus 1 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
                  work_dir="${DATA_DIR}"