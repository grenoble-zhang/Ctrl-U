<div align ="center">
<h1> Ctrl-U </h1>
<h3> Robust Conditional Image Generation via Uncertainty-aware Reward Modeling </h3>
<div align="center">
</div>

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://grenoble-zhang.github.io/Ctrl-U-Page/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2410.11236-b31b1b.svg)](https://arxiv.org/abs/2410.11236)&nbsp;
</div>

Authors: [Guiyu Zhang\*](https://scholar.google.com/citations?user=NLPMoeAAAAAJ/)<sup>1,2</sup>, [Huan-ang Gao\*](https://c7w.tech/about/)<sup>2</sup>, Zijian Jiang<sup>2</sup>, [Hao Zhao†](https://sites.google.com/view/fromandto)<sup>2</sup>, [Zhedong Zheng†](https://www.zdzheng.xyz/)<sup>1</sup>

<sup>1</sup> FST, University of Macau&emsp;<sup>2</sup> AIR, Tsinghua University

## News

`[2025-2-19]:` The code and models have been released 😊!

`[2025-1-22]:` Our Ctrl-U has been accepted by ICLR 2025 🎉 !

`[2024-10-14]:` We have released the [technical report of Ctrl-U](https://arxiv.org/abs/2410.11236).

## Getting Started
### 🛠️ Environments
```bash
git clone https://github.com/grenoble-zhang/Ctrl-U.git
cd Ctrl-U
conda create -n Ctrl-U python=3.10
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip3 install "mmsegmentation>=1.0.0"
pip3 install mmdet
```

### 🕹️ Data Preperation
**All the organized data has been uploaded to Hugging Face and will be automatically downloaded during training or evaluation.** You can preview it in advance using the following links to check the data samples and the disk space required.




|   Task    | Training Data 🤗 | Evaluation Data 🤗 |
|:----------:|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
|  LineArt, Hed  | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_train), 1.14 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_canny_eval), 2.25GB |
|  Depth   |  [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth), 1.22 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth_eval), 2.17GB |
|  Segmentation ADE20K   | [Data](https://huggingface.co/datasets/limingcv/Captioned_ADE20K), 7.04 GB | Same Path as Training Data |
|  Segmentation COCOStuff   | [Data](https://huggingface.co/datasets/limingcv/Captioned_ADE20K), 61.9 GB | Same Path as Training Data |


### 😉 Training

```bash
bash train/ctrlu_ade20k.sh
bash train/ctrlu_cocostuff.sh
bash train/ctrlu_depth.sh
bash train/ctrlu_hed.sh
bash train/ctrlu_lineart.sh
```

### 🧐 Evaluation
Please download the model weights and put them into each subset of `checkpoints`:
|   model    |HF weights                                                                        |
|:----------:|:------------------------------------------------------------------------------------|
|  Segmentation_ade20k   | [model](https://huggingface.co/grenoble/Ctrl-u/tree/main/checkpoint/reward/ade20k) |
|  Segmentation_cocostuff   | [model](https://huggingface.co/grenoble/Ctrl-u/tree/main/checkpoint/reward/cocostuff) |
|  Depth   |  [model](https://huggingface.co/grenoble/Ctrl-u/tree/main/checkpoint/reward/depth) |
|  Hed (SoftEdge)   | [model](https://huggingface.co/grenoble/Ctrl-u/tree/main/checkpoint/reward/hed) |
|  LineArt   | [model](https://huggingface.co/grenoble/Ctrl-u/tree/main/checkpoint/reward/lineart) |

Please make sure the folder directory is consistent with the test script, then you can eval each model by:
```bash
bash eval/eval_ade20k.sh
bash eval/eval_cocostuff.sh
bash eval/eval_depth.sh
bash eval/eval_hed.sh
bash eval/eval_lineart.sh
```
Please refer to the code for evaluating [CLIP-Score](eval/eval_clip.py) and [FID](eval/eval_fid.py)

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments
Our work is based on the following open-source projects. We sincerely thank the contributors for thoese great works!
* [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

## Citation
If you find Ctrl-U is useful in your research or applications, please consider giving us a star ⭐ or cite us using:
```bibtex
@article{zhang2024ctrl,
  title={Ctrl-U: Robust Conditional Image Generation via Uncertainty-aware Reward Modeling},
  author={Zhang, Guiyu and Gao, Huan-ang and Jiang, Zijian and Zhao, Hao and Zheng, Zhedong},
  journal={arXiv preprint arXiv:2410.11236},
  year={2024}
}
```
