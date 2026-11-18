# coding=utf-8
# Copyright 2025 Harutatsu Akiyama and HuggingFace Inc.
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

import random

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)

from ...testing_utils import floats_tensor, torch_device
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusionXLInstructPix2PixPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLInstructPix2PixPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
    }
    batch_input_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 5 * 8 + 32
            cross_attention_dim=64,
        )

        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
        }
        return components

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "image_guidance_scale": 1,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionXLInstructPix2PixPipeline(
    StableDiffusionXLInstructPix2PixPipelineTesterConfig, PipelineTesterMixin
):
    def test_stable_diffusion_xl_pix2pix_custom_timesteps(self):
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        default_image = sd_pipe(**inputs).images

        # Timesteps matching the default 2-step schedule must reproduce the default output.
        default_timesteps = sd_pipe.scheduler.timesteps.tolist()
        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = default_timesteps
        timesteps_image = sd_pipe(**inputs).images
        assert timesteps_image.shape == (1, 3, 64, 64)
        assert torch.allclose(timesteps_image, default_image, atol=1e-4, rtol=1e-4)

        # A different schedule should change the output.
        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [900, 200]
        custom_image = sd_pipe(**inputs).images
        assert (custom_image - default_image).abs().max() > 1e-3

    def test_stable_diffusion_xl_pix2pix_custom_sigmas(self):
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        default_image = sd_pipe(**inputs).images

        # Sigmas matching the default 2-step schedule must reproduce the default output.
        default_sigmas = sd_pipe.scheduler.sigmas.tolist()
        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["sigmas"] = default_sigmas
        sigmas_image = sd_pipe(**inputs).images
        assert sigmas_image.shape == (1, 3, 64, 64)
        assert torch.allclose(sigmas_image, default_image, atol=1e-4, rtol=1e-4)

    def test_stable_diffusion_xl_pix2pix_timesteps_sigmas_mutually_exclusive(self):
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [900, 200]
        inputs["sigmas"] = [7.0, 1.0, 0.0]
        with pytest.raises(ValueError, match="Only one of `timesteps` or `sigmas`"):
            sd_pipe(**inputs)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    @pytest.mark.skip("Every `_optional_component` is needed to encode the prompt this pipeline requires.")
    def test_save_load_optional_components(self):
        pass


class TestStableDiffusionXLInstructPix2PixPipelineMemory(
    StableDiffusionXLInstructPix2PixPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL pix2pix pipeline."""
