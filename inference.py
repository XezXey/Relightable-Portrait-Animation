#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import os
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, EulerDiscreteScheduler

from src.modules.head_net import HeadNet
from src.modules.light_net import LightNet
from src.modules.ref_net import RefNet
from src.modules.unet import UNetSpatioTemporalConditionModel
from src.pipelines.pipeline_relightalbepa_composer import RelightablepaPipeline


class RelightablePA():
    def __init__(self, pretrained_model_name_or_path, checkpoint_path, weight_dtype=torch.float16, device="cuda"):
        # Load scheduler, tokenizer and models.
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path, subfolder="feature_extractor" 
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="sd-vae-ft-mse")
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        self.head_embedder = HeadNet(noise_latent_channels=320)
        self.light_embedder = LightNet(noise_latent_channels=320)
        self.ref_embedder = RefNet(noise_latent_channels=320)

        # Freeze vae and image_encoder
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.head_embedder.requires_grad_(False)
        self.light_embedder.requires_grad_(False)
        self.ref_embedder.requires_grad_(False)

        self.unet.load_state_dict(torch.load(f"{checkpoint_path}/unet.pth"))
        self.head_embedder.load_state_dict(torch.load(f"{checkpoint_path}/head_embedder.pth"))
        self.light_embedder.load_state_dict(torch.load(f"{checkpoint_path}/light_embedder.pth"))
        self.ref_embedder.load_state_dict(torch.load(f"{checkpoint_path}/ref_embedder.pth"))

        self.weight_dtype = weight_dtype
        self.device = device

        self.image_encoder.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.unet.to(device, dtype=weight_dtype)
        self.head_embedder.to(device, dtype=weight_dtype) 
        self.light_embedder.to(device, dtype=weight_dtype) 
        self.ref_embedder.to(device, dtype=weight_dtype)

        # The models need unwrapping because for compatibility in distributed training mode.
        self.pipeline = RelightablepaPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=self.unet,
            image_encoder=self.image_encoder,
            vae=self.vae,
            head_embedder=self.head_embedder,
            light_embedder=self.light_embedder,
            ref_embedder=self.ref_embedder,
            torch_dtype=weight_dtype,
        )
        self.pipeline = self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=False)

    def portrait_animation_and_relighting(self, video_path, save_path, guidance, inference_steps, driving_mode="relighting"):
        path = "resources/target/"
        path_tmp = "resources/tmp/"
        if not os.path.exists(path):
            os.system(f"mkdir {path}")
        else:
            os.system(f"rm -r {path}/*")

        if not os.path.exists(path_tmp):
            os.system(f"mkdir {path_tmp}")
        else:
            os.system(f"rm -r {path_tmp}/*")
        
        os.system(f"ffmpeg -i {video_path} {path}/%5d.png")
        
        pixel_values = []
        pixel_head = []
        pixel_values_light = []
        img = np.array(Image.open(path + "00001.png"))
        # img = cv2.resize(img, (img.shape[1], img.shape[0]))
        pixel_ref_values = img[:, :512]
        pixel_ref_mask = img[:, 512:1024]
        pixel_ref_mask = cv2.resize(pixel_ref_mask, (64, 64))
        # pixel_ref_mask = np.ones_like(pixel_ref_mask) * 255

        for i in range(1, len(os.listdir(path))+1):
            img = np.array(Image.open(f"{path}/{str(i).zfill(5)}.png"), dtype=np.uint8)
            # img = cv2.resize(img, (img.shape[1], img.shape[0]))
            pixel_values.append(img[:, 1024:1536][None])
            pixel_head.append(img[:, 1536:2048][None])
            pixel_values_light.append(img[:, 2048:2560][None])

        pixel_values = torch.tensor(np.concatenate(pixel_values, axis=0)[None]).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 127.5 - 1.0
        pixel_head = torch.tensor(np.concatenate(pixel_head, axis=0)[None]).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 255.0
        pixel_values_light = torch.tensor(np.concatenate(pixel_values_light, axis=0)[None]).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 255.0

        pixel_ref_values = torch.tensor(pixel_ref_values[None, None]).repeat(1, pixel_values.size(1), 1, 1, 1).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 127.5 - 1.0
        pixel_ref_mask = torch.tensor(pixel_ref_mask[None, None]).repeat(1, pixel_values.size(1), 1, 1, 1).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3)[:, :, 0:1] / 255.0

        num_frames = pixel_values.size(1)
        pixel_pil = [Image.fromarray(np.uint8((pixel_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i] + 1) * 127.5)) for i in range(num_frames)]
        heads_pil = [Image.fromarray(np.uint8((pixel_head.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i]) * 255)) for i in range(num_frames)]
        lights_drv_pil = [Image.fromarray(np.uint8((pixel_values_light.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i]) * 255)) for i in range(num_frames)]
        reference_pil = [Image.fromarray(np.uint8((pixel_ref_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, 0] + 1) * 127.5))]

        if driving_mode == "relighting":
            model_args = [{"image_head": None, "image_light": pixel_values_light, "image_ref": pixel_ref_values}, # cond
                        {"image_head": None, "image_light": None,               "image_ref": pixel_ref_values}] # uncond
        elif driving_mode == "landmark":
            model_args = [{"image_head": pixel_head, "image_light": None, "image_ref": pixel_ref_values},         # cond
                        {"image_head": None, "image_light": None,               "image_ref": None}]             # uncond
        else:
            model_args = [{"image_head": None, "image_light": pixel_values_light, "image_ref": pixel_ref_values}, # cond
                        {"image_head": None, "image_light": None,               "image_ref": None}]             # uncond
            
        frames = self.pipeline(
            reference_pil, model_args=model_args, image_mask=pixel_ref_mask, 
            num_frames=pixel_head.size(1),
            tile_size=16, tile_overlap=6,
            height=512, width=512, fps=7,
            noise_aug_strength=0.02, num_inference_steps=inference_steps,
            generator=None, min_guidance_scale=guidance, 
            max_guidance_scale=guidance, decode_chunk_size=8, output_type="pt", device="cuda"
        ).frames.cpu()
        video_frames = (frames.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).numpy()[0]

        final = []
        for i in range(pixel_head.size(1)):
            img = video_frames[i]
            head = np.array(heads_pil[i])
            light = np.array(lights_drv_pil[i])
            tar = np.array(pixel_pil[i])
            ref = np.array(reference_pil[0])
            # final.append(np.concatenate([ref, head, light, img, tar], axis=1))
            Image.fromarray(np.uint8(np.concatenate([ref, light, img, tar], axis=1))).save(f"{path_tmp}/{str(i).zfill(5)}.png")

        os.system(f"ffmpeg -r 20 -i {path_tmp}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")
        # torchvision.io.write_video(save_path, final, fps=20, video_codec='h264', options={'crf': '10'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pretrained_weights/stable-video-diffusion-img2vid-xt", help="stable video diffusion pretrained model path") 
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_weights/relipa", help="relightable portrait animation checkpoint path") 
    parser.add_argument("--video_path", type=str, default="resources/shading.mp4", help="reference and shading") 
    parser.add_argument("--save_path", type=str, default="result.mp4", help="result save path")
    parser.add_argument("--guidance", type=float, default=4.5, help="lighting intensity")
    parser.add_argument("--inference_steps", type=int, default=25, help="diffusion reverse sampling steps")
    parser.add_argument("--driving_mode", type=str, default="relighting", help="relighting | landmark")

    args = parser.parse_args()

    relightablepa = RelightablePA(pretrained_model_name_or_path=args.pretrained_model_name_or_path, checkpoint_path=args.checkpoint_path)
    relightablepa.portrait_animation_and_relighting(video_path=args.video_path, 
                                                    save_path=args.save_path, 
                                                    guidance=args.guidance, 
                                                    inference_steps=args.inference_steps, 
                                                    driving_mode=args.driving_mode)
         
         
