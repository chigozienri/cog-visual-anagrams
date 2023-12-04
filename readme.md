This is a cog wrapper for the following paper:


# Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models

[Daniel Geng](https://dangeng.github.io/), [Aaron Park](https://inbumpark.github.io/), [Andrew Owens](https://andrewowens.com/)

## [[Arxiv](https://arxiv.org/abs/2311.17919)] [[Website](https://dangeng.github.io/visual_anagrams/)] [[Colab](https://colab.research.google.com/drive/1hCvJR5GsQrhH1ceDjdbzLG8y6m2UdJ6l?usp=sharing)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hCvJR5GsQrhH1ceDjdbzLG8y6m2UdJ6l?usp=sharing)

![teaser](./assets/teaser.small.gif)

This repo contains code to generate visual anagrams and other multi-view optical illusions. These are images that change appearance or identity when transformed, such as by a rotation, a color inversion, or a jigsaw rearrangement. Please read our paper or visit our website for more details.

## Usage

```bash
cog predict \
  -i 'seed=0' \
  -i 'style="an oil painting of "' \
  -i 'video=true' \
  -i 'views="identity, jigsaw"' \
  -i 'prompts="a rabbit, a coffee cup"' \
  -i 'num_samples=1' \
  -i 'guidance_scale_1=10' \
  -i 'guidance_scale_2=10' \
  -i 'num_inference_steps_1=30' \
  -i 'num_inference_steps_2=30'
```

## The Art of Choosing Prompts

Choosing prompts for illusions can be fairly tricky and unintuitive. Here are some tips:

- Intuition and reasoning works less often than you would expect. Prompts that you think would work great often work poorly, and vice versa. So exploration is key.
- Styles such as `"a photo of"` tend to be harder as the constraint of realism is fairly difficult (but this doesn't mean they can't work!).
- Conversely, styles such as `"an oil painting of"` seem to do better because there's more freedom to how it can be depicted and interpreted.
- In a similar vein, subjects that allow for high degrees of flexibility in depiction tend to be good. For example, prompts such as `"houseplants"` or `"wine and cheese"` or `"a kitchen"`
- But be careful the subject is still easily recognizable. Illusions are much better when they are instantly understandable.
- Faces often make for very good "hidden" subjects. This is probably because the human visual system is particularly adept at picking out faces. For example, `"an old man"` or `"marilyn monroe"` tend to be good subjects.
- Perhaps a bit evident, but 3 view and 4 view illusions are considerably more difficult to get to work.
