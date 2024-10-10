DATA_DIR=$1
GENERATED=$1
NUM_GPUS=1
export CLIP_DATASET="/Node10_nvme/MultiGen-20M_canny_eval/data"
export FID_DATASET="/Node09_nvme/zhanggy/CPP_Uncertainty/work_dirs/MultiGen-20M_canny_eval"
export HF_ENDPOINT=https://hf-mirror.com

python3 eval/eval_edge.py --task lineart --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS} > $GENERATED/metrics.txt 2>&1



# python3 eval/clip_fid_text.py \
#     --generated=${GENERATED} \
#     --clip_dataset=${CLIP_DATASET} \
#     --only_image \
#     --fid_dataset=${FID_DATASET} > $GENERATED/clip_fid.txt 2>&1