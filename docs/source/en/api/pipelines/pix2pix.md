<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# InstructPix2Pix

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800) is by Tim Brooks, Aleksander Holynski and Alexei A. Efros.

The abstract from the paper is:

*We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models -- a language model (GPT-3) and a text-to-image model (Stable Diffusion) -- to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.*

You can find additional information about InstructPix2Pix in the [paper](https://huggingface.co/papers/2211.09800), [original codebase](https://github.com/timothybrooks/instruct-pix2pix), and try it out in a [demo](https://huggingface.co/spaces/timbrooks/instruct-pix2pix).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

## Custom noise schedules

Both InstructPix2Pix pipelines accept a pluggable scheduler and let you override the default noise schedule with custom
`timesteps` or `sigmas`. This makes it possible to shorten the edit to a few steps or to use a curated schedule (such as
the [Align Your Steps](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) schedule) without changing the
checkpoint.

Swap the scheduler with [`~ConfigMixin.from_config`] so the new scheduler keeps the trained `num_train_timesteps` and
related settings:

```py
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerDiscreteScheduler
from diffusers.utils import load_image

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

image = load_image(
    "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
).resize((512, 512))

edited_image = pipe(
    prompt="make the mountains snowy",
    image=image,
    timesteps=[999, 750, 500, 250],
).images[0]
```

`timesteps` must be in descending order. You can pass `sigmas` instead (the last sigma should be `0.0`), but only one
of the two can be used at a time. Both arguments only work with schedulers whose `set_timesteps` accepts them -
otherwise the pipeline raises a `ValueError` telling you to pick a compatible scheduler. See the Schedulers
[guide](../../using-diffusers/schedulers#timestep-schedules) for the full list of supported options and the
[AYS schedules](../../using-diffusers/schedulers#sigmas) example.

## StableDiffusionInstructPix2PixPipeline
[[autodoc]] StableDiffusionInstructPix2PixPipeline
	- __call__
	- all
	- load_textual_inversion
	- load_lora_weights
	- save_lora_weights

## StableDiffusionXLInstructPix2PixPipeline
[[autodoc]] StableDiffusionXLInstructPix2PixPipeline
	- __call__
	- all
