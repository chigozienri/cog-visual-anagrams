import os
import tempfile
from typing import Iterator

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
        style: str = Input(),
        prompts: str = Input(description="Comma-separated list of prompts"),
        views: str = Input(
            description="Comma-separated list of views, e.g. `jigsaw`. Must be same length as prompts"
        ),
        num_samples: int = Input(default=1),
        num_inference_steps_1: int = Input(default=30),
        guidance_scale_1: float = Input(default=10.0),
        num_inference_steps_2: int = Input(default=30),
        guidance_scale_2: float = Input(default=10.0),
        noise_level: int = Input(default=50),
        seed: int = Input(default=0),
        video: bool = Input(default=True),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        prompts = [prompt.strip() for prompt in prompts.split(",")]
        views = [view.strip() for view in views.split(",")]
        prompt_embeds = [
            self.stage_1.encode_prompt(f"{style} {p}".strip()) for p in prompts
        ]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)
        views = get_views(views)

        save_dir = Path(tempfile.mkdtemp())

        # # Save metadata
        # save_metadata(views, args, save_dir)
        if video and len(views) != 2 and views[0] != "identity":
            print(
                "WARNING: Outputting a video requires only two views, and the first view must be the identity. This run will not output a video."
            )
            video = False
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
                noise_level=noise_level,
                generator=torch.manual_seed(seed + i),
            )
            save_illusion(image, views, sample_dir)

            print([f for f in os.listdir(sample_dir)])
            for j, f in enumerate(os.listdir(sample_dir)):
                im_path = os.path.join(sample_dir, f)
                yield Path(im_path)
                if video and j == 0:
                    video_path = os.path.join(sample_dir, "video.mp4")
                    animate_two_view(
                        im_path,
                        views[1],
                        style + prompts[0],
                        style + prompts[1],
                        save_video_path=video_path,
                        hold_duration=120,
                        text_fade_duration=10,
                        transition_duration=45,
                        im_size=256,
                        frame_size=384,
                    )

                    yield (Path(video_path))
