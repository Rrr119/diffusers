#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""InstructPix2Pix image editing with a pluggable scheduler and a custom noise schedule.

Example:

```bash
python run_custom_schedule.py \
    --model_id timbrooks/instruct-pix2pix \
    --image_url https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png \
    --prompt "make the mountains snowy" \
    --scheduler euler \
    --timesteps 999 750 500 250 \
    --output edited.png
```
"""

import argparse

import torch

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.utils import load_image


SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--image_url", required=True, help="URL or local path of the image to edit.")
    parser.add_argument("--prompt", required=True, help="Edit instruction.")
    parser.add_argument(
        "--scheduler",
        choices=sorted(SCHEDULERS.keys()),
        default="euler",
        help="Scheduler to swap in. Both choices support custom timesteps and sigmas.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Custom descending timesteps, e.g. `--timesteps 999 750 500 250`.",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Custom sigmas (the last value should be 0.0). Mutually exclusive with `--timesteps`.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--output", default="edited.png")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timesteps is not None and args.sigmas is not None:
        raise ValueError("Pass only one of `--timesteps` or `--sigmas`.")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    # Swap the scheduler while keeping the checkpoint's scheduler config.
    pipe.scheduler = SCHEDULERS[args.scheduler].from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = load_image(args.image_url).resize((512, 512))

    call_kwargs = {
        "prompt": args.prompt,
        "image": image,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
    }
    if args.timesteps is not None:
        call_kwargs["timesteps"] = args.timesteps
    elif args.sigmas is not None:
        call_kwargs["sigmas"] = args.sigmas
    else:
        call_kwargs["num_inference_steps"] = args.num_inference_steps

    edited_image = pipe(**call_kwargs).images[0]
    edited_image.save(args.output)
    print(f"Saved edited image to {args.output}")


if __name__ == "__main__":
    main()
