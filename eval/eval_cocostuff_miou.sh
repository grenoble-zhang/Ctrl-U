DATA_DIR=$1
GENERATED=$1
export CLIP_DATASET="/Node10_nvme/Captioned_COCOStuff/data"
export FID_DATASET="/Node09_nvme/zhanggy/CPP_Uncertainty/work_dirs/Captioned_COCOStuff_val"
export HF_ENDPOINT=https://hf-mirror.com

# mim test mmseg mmlab/mmseg/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py \
#     --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth \
#     --gpus 1 \
#     --launcher pytorch \
#     --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
#                   test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
#                   work_dir="${DATA_DIR}" > $DATA_DIR/miou.txt 2>&1

python3 eval/clip_fid.py \
    --generated=${GENERATED} \
    --clip_dataset=${CLIP_DATASET} \
    --fid_dataset=${FID_DATASET} > $GENERATED/clip_fid.txt 2>&1