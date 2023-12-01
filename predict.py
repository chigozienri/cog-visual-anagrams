import os
import tempfile
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline

from animate import animate_two_view
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
from visual_anagrams.views import get_views
from weights_downloader import WeightsDownloader

MODEL_CACHE = "diffusers-cache"
DEEPFLOYD_IF_URL = (
    "https://weights.replicate.delivery/default/deep-floyd/DeepFloyd--IF-I-M-v1.0.tar"
)


class Predictor(BasePredictor):
    def setup(self):
        """Load the models into memory to make running multiple predictions efficient"""

        WeightsDownloader.download_if_not_exists(DEEPFLOYD_IF_URL, MODEL_CACHE)
        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=MODEL_CACHE,
        )
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-M-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=MODEL_CACHE,
        )
        # TODO: CPU offloading is breaking num_samples > 1
        # self.stage_1.enable_model_cpu_offload()
        # self.stage_2.enable_model_cpu_offload()
        self.stage_1 = self.stage_1.to("cuda")
        self.stage_2 = self.stage_2.to("cuda")

    # Available Views:
    # 'identity', 'flip', 'rotate_cw', 'rotate_ccw', 'rotate_180', 'negate', 'skew',
    # 'patch_permute', 'pixel_permute', 'jigsaw', 'inner_circle'

    def predict(
        self,
        style: str = Input(
            description="Prompt prefix of the visual style. Looser styles work better.",
            default="an oil painting of ",
        ),
        prompts: str = Input(
            description="Comma-separated list of prompts",
            default="a rabbit, a coffee cup",
        ),
        views: str = Input(
            description=("Comma-separated list of views. Must be same length as prompts. "
                         "First view should usually be 'identity'. Available Views: 'identity', "
                         "'flip', 'rotate_cw', 'rotate_ccw', 'rotate_180', 'negate', 'skew', "
                         "'patch_permute', 'pixel_permute', 'jigsaw', 'inner_circle'"),
            default="identity, jigsaw",
        ),
        num_samples: int = Input(default=1),
        num_inference_steps_1: int = Input(default=30),
        guidance_scale_1: float = Input(default=10.0),
        num_inference_steps_2: int = Input(default=30),
        guidance_scale_2: float = Input(default=10.0),
        seed: int = Input(default=None, description="Leave empty for a random seed"),
        video: bool = Input(default=True),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        prompts = [prompt.strip() for prompt in prompts.split(",")]
        prompts = [style.strip() + (" " if len(style) > 0 else "") + prompt for prompt in prompts]
        views = [view.strip() for view in views.split(",")]
        prompt_embeds = [
            self.stage_1.encode_prompt(p) for p in prompts
        ]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)
        views = get_views(views)

        save_dir = Path(tempfile.mkdtemp())

        # # Save metadata
        # save_metadata(views, args, save_dir)
        if video and len(views) != 2 and (views[0] != "identity" or views[1] in "pixel_permute, patch_permute"):
            print(
                "WARNING: Outputting a video requires only two views, and the first view must be the identity. "
                "pixel_permute and patch_permute also don't support video yet. This run will not output a video."
            )
            video = False
        outputs = []
        for i in range(num_samples):
            sample_dir = save_dir / f"{i:04}"
            sample_dir.mkdir(exist_ok=True, parents=True)
            image = sample_stage_1(
                self.stage_1,
                prompt_embeds,
                negative_prompt_embeds,
                views,
                num_inference_steps=num_inference_steps_1,
                guidance_scale=guidance_scale_1,
                reduction="mean",
                generator=torch.manual_seed(seed + i),
            )
            image = sample_stage_2(
                self.stage_2,
                image,
                prompt_embeds,
                negative_prompt_embeds,
                views,
                num_inference_steps=num_inference_steps_2,
                guidance_scale=guidance_scale_2,
                reduction="mean",
                noise_level=50,
                generator=torch.manual_seed(seed + i),
            )
            save_illusion(image, views, sample_dir)

            print([f for f in os.listdir(sample_dir)])
            for f in os.listdir(sample_dir):
                im_path = os.path.join(sample_dir, f)
                outputs.append(Path(im_path))
            if video:
                video_path = os.path.join(sample_dir, "video.mp4")
                animate_two_view(
                    os.path.join(sample_dir, os.listdir(sample_dir)[0]),
                    views[1],
                    prompts[0],
                    prompts[1],
                    save_video_path=video_path,
                    hold_duration=120,
                    text_fade_duration=10,
                    transition_duration=45,
                    im_size=256,
                    frame_size=384,
                )

                outputs.append(Path(video_path))
        return outputs
