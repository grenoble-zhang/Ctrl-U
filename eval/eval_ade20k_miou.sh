DATA_DIR=$1
GENERATED=$1
export CLIP_DATASET="/Node10_nvme/Captioned_ADE20K/data"
export FID_DATASET="/Node09_nvme/zhanggy/CPP_Uncertainty/work_dirs/Captioned_ADE20K/image"
export HF_ENDPOINT=https://hf-mirror.com

# mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
#     --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
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