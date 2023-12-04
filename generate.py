import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
from visual_anagrams.views import get_views

# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# Do admin stuff
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# Make models
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16
)
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-M-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
)
stage_1.enable_model_cpu_offload()
stage_2.enable_model_cpu_offload()
stage_1 = stage_1.to(args.device)
stage_2 = stage_2.to(args.device)

# Get prompt embeddings
prompt_embeds = [
    stage_1.encode_prompt(f"{args.style} {p}".strip()) for p in args.prompts
]
prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
prompt_embeds = torch.cat(prompt_embeds)
negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

# Get views
views = get_views(args.views)

# Save metadata
save_metadata(views, args, save_dir)

# Sample illusions
for i in range(args.num_samples):
    # Admin stuff
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f"{i:04}"
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Sample 64x64 image
    image = sample_stage_1(
        stage_1,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        reduction=args.reduction,
        generator=generator,
    )
    save_illusion(image, views, sample_dir)

    # Sample 256x256 image, by upsampling 64x64 image
    image = sample_stage_2(
        stage_2,
        image,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        reduction=args.reduction,
        noise_level=args.noise_level,
        generator=generator,
    )
    save_illusion(image, views, sample_dir)
