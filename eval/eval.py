import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import random
import torch
import numpy as np
import torchvision.transforms.functional as F
import argparse
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel,
    UniPCMultistepScheduler, DDIMScheduler,
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline, AutoencoderKL
)
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from datasets import load_dataset, load_from_disk
from accelerate import PartialState
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from torchvision.transforms import Compose, Normalize, ToTensor
transforms = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None



def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_noise(seed, latent):
    generator = torch.manual_seed(seed)
    return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu").to(latent.dtype)

def max_denoise(model_sampling, sigmas):
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

def preprocess_image(image, batch_size):
    w, h = image.size
    w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def main(args):
    distributed_state = PartialState()
    seed_torch(args.seed)

    # load_dataset
    if args.dataset_name.count('/') == 1:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            split=args.dataset_split
        )
    else:
         # Loading from local disk.
        # dataset = load_from_disk(
        #     dataset_path=args.dataset_name,
        #     split=args.dataset_split
        # )
        dataset = load_dataset('parquet', data_files=f'{args.dataset_name}/validation-*.parquet')["train"]
        # dataset = load_dataset('parquet', data_files=f'{args.dataset_name}/test-*.parquet')["train"]

    print(f"Loading pre-trained weights from {args.model_path}")

    # main_process_first: Avoid repeated downloading of models for all processes
    with distributed_state.main_process_first():
        # load pre-trained model
        if args.model == 'controlnet':
            controlnet = ControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=args.sd_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
            )
        elif args.model == 'controlnet-sdxl':
            controlnet = ControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                args.sd_path,  # "stabilityai/stable-diffusion-xl-base-1.0"
                controlnet=controlnet,
                vae=vae,
                torch_dtype=torch.float16,
            )
        elif args.model == 't2i-adapter':
            adapter = T2IAdapter.from_pretrained(args.model_path, torch_dtype=torch.float16)
            pipe = StableDiffusionAdapterPipeline.from_pretrained(
                args.sd_path, adapter=adapter, safety_checker=None, torch_dtype=torch.float16, variant="fp16"
            )
        elif args.model == 't2i-adapter-sdxl':
            adapter = T2IAdapter.from_pretrained(args.model_path, torch_dtype=torch.float16, varient="fp16")
            euler_a = EulerAncestralDiscreteScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
            vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                args.sd_path, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented")

    pipe.to(distributed_state.device)

    with distributed_state.main_process_first():
        if args.task_name == 'depth':
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        elif args.task_name == 'lineart':
            from utils import get_reward_model
            model = get_reward_model(task='lineart', model_path='https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth')
            model.eval()
        elif args.task_name == 'hed':
            from utils import get_reward_model
            model = get_reward_model(task='hed', model_path='https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth')
            model.eval()

    # only the main process will create the output directory
    save_dir = os.path.join(args.output_dir, args.dataset_name.split('/')[-1], args.dataset_split, args.exp_name)
    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "images"))
            os.makedirs(os.path.join(save_dir, "annotations"))
            os.makedirs(os.path.join(save_dir, "visualization"))

        for i in range(args.batch_size):
            if not os.path.exists(os.path.join(save_dir, f"images/group_{i}")):
                os.makedirs(os.path.join(save_dir, f"images/group_{i}"))
            if args.task_name in ['lineart', 'hed', 'depth']:
                if not os.path.exists(os.path.join(save_dir, f"images/only_image_group_{i}")):
                    os.makedirs(os.path.join(save_dir, f"images/only_image_group_{i}"))
    if args.model == 'controlnet':
        if args.ddim:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # NOTE: assign a specific gpu_id is necessary, otherwise all models will be loaded on gpu 0
    pipe.enable_model_cpu_offload(gpu_id=distributed_state.process_index)

    if distributed_state.is_main_process:
        start_time = time.time()

    # split dataset into multiple processes and gpus, each process corresponds to a gpu
    with distributed_state.split_between_processes(list(range(len(dataset)))) as local_idxs:

        print(f"{distributed_state.process_index} has {len(local_idxs)} images")

        for idx in local_idxs:
            # Unique Identifier, used for saving images while avoid overwriting due to the same prompt
            print(f"Processing image {idx}...")
            uid = str(idx)
            if os.path.exists(f'{save_dir}/visualization/{uid}.png'):
                continue

            original_image = dataset[idx][args.image_column].convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            condition = dataset[idx][args.condition_column].convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            prompt = dataset[idx][args.prompt_column]
            label = dataset[idx][args.label_column] if args.label_column is not None else None

            if args.task_name in ['lineart', 'hed']:
                condition = F.pil_to_tensor(condition.resize((args.resolution, args.resolution))).unsqueeze(0) / 255.0
                with torch.no_grad():
                    condition = model(condition)

                condition = 1 - condition if args.task_name == 'lineart' else condition
                condition = condition.reshape(args.resolution, args.resolution)
                condition = F.to_pil_image(condition, 'L').convert('RGB')
                label = condition

            image = original_image.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            condition = condition.resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
            prompts, conditions = [prompt] * args.batch_size, [condition] * args.batch_size

            if args.model == 't2i-adapter-sdxl' and args.task_name == 'lineart':
                images = pipe(
                    prompt=prompts,
                    image=conditions,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    adapter_conditioning_scale=0.5,
                    negative_prompt=['worst quality, low quality'] *  args.batch_size
                ).images
            else:
                images = pipe(
                    prompt=prompts,
                    image=conditions,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=['worst quality, low quality'] *  args.batch_size
                ).images

            if args.task_name in ['lineart', 'hed']:
                lineart = [F.pil_to_tensor(img)/255.0 for img in images]
                with torch.no_grad():
                    lineart = model(torch.stack(lineart))
                lineart = torch.chunk(lineart, args.batch_size, dim=0)
                lineart = [x.reshape(1, args.resolution, args.resolution) for x in lineart]
                lineart = [F.to_pil_image(x).convert('RGB') for x in lineart]
                [img.save(f"{save_dir}/images/group_{i}/{uid}_lineart.png") for i, img in enumerate(lineart)]
            elif args.task_name == 'depth':
                label = np.array(label)
                label = (label - label.min()) / (label.max() - label.min())
                label = label * 255
                depth_model_input = processor(images=images, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**depth_model_input)
                    predicted_depth = outputs.predicted_depth
                    depth_maps = [F.to_pil_image(x/x.max()).convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BILINEAR) for x in predicted_depth]
                    [img.convert('L').save(f"{save_dir}/images/group_{i}/{uid}_depth.png") for i, img in enumerate(depth_maps)]

            # save ground truth labels
            if label is not None:
                label = Image.fromarray(np.array(label).astype('uint8'))
                label.resize((args.resolution, args.resolution), Image.Resampling.NEAREST).save(f"{save_dir}/annotations/{uid}.png")
            # scale the generated images to the original resolution for evaluation
            # then save the generated images for evaluation
            [img.save(f"{save_dir}/images/group_{i}/{uid}.png") for i, img in enumerate(images)]
            if args.task_name in ['lineart', 'hed', 'depth']:
                pass
                # [img.save(f"{save_dir}/images/only_image_group_{i}/{uid}.png") for i, img in enumerate(images)]
            # generate a grid of images
            if args.task_name in ['lineart', 'hed']:
                # input image, condition image, generated_images
                if args.task_name == 'lineart':
                    condition = 255 - F.pil_to_tensor(condition)
                    condition = F.to_pil_image(condition)

                images = [image] + images + [condition] + lineart
                images = [img.convert('RGB') for img in images] if args.model == 't2i-adapter' else images
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            elif args.task_name == 'depth':
                # input image, condition image, generated_images
                images = [image] + images + [condition] + depth_maps
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            else:
                # input image, condition image, generated_images
                images = [image] + [condition] + images
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images))

            show(images)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/visualization/{uid}.png', bbox_inches='tight', dpi=512)
            plt.clf()

    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        end_time = time.time()
        print(f"Validation time: {end_time - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Image Generation')
    parser.add_argument('--task_name', type=str, default='seg')
    parser.add_argument('--dataset_name', type=str, default='limingcv/Captioned_COCOStuff', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='lllyasviel/control_v11p_sd15_seg')
    parser.add_argument('--sd_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--image_column', type=str, default='image')
    parser.add_argument('--condition_column', type=str, default='control_seg')
    parser.add_argument('--label_column', type=str, default=None)
    parser.add_argument('--prompt_column', type=str, default='prompt')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for image generation')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the image')
    parser.add_argument('--output_dir', type=str, default='work_dirs/eval_dirs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default="data/huggingface_datasets", help='Cache directory for dataset and models')
    parser.add_argument('--model', type=str, default="controlnet")
    parser.add_argument('--ddim', action='store_true', help='weather use DDIM instead of UniPC')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    args = parser.parse_args()
    main(args)
