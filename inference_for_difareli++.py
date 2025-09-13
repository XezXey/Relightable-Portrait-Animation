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

import os, glob
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
import torchvision
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, EulerDiscreteScheduler

from src.modules.head_net import HeadNet
from src.modules.light_net import LightNet
from src.modules.ref_net import RefNet
from src.modules.unet import UNetSpatioTemporalConditionModel
from src.pipelines.pipeline_relightalbepa_composer import RelightablepaPipeline
import tqdm, json, subprocess
from utils.logging import createLogger
import warnings
warnings.filterwarnings('ignore')



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

    def portrait_animation_and_relighting(self, video_path, save_path, guidance, inference_steps, save_vid=True, driving_mode="relighting", eval_dict=None):
        os.makedirs(save_path, exist_ok=True)
        pixel_values = []
        pixel_head = []
        pixel_values_light = []
        img = np.array(Image.open(video_path + "00001.png"))
        # img = cv2.resize(img, (img.shape[1], img.shape[0]))
        pixel_ref_values = img[:, :512]
        pixel_ref_mask = img[:, 512:1024]
        pixel_ref_mask = cv2.resize(pixel_ref_mask, (64, 64))
        # pixel_ref_mask = np.ones_like(pixel_ref_mask) * 255

        # for i in range(1, len(glob.glob(f"{video_path}/*.png"))+1):
        for i in range(len(glob.glob(f"{video_path}/*.png"))):
            img = np.array(Image.open(f"{video_path}/{str(i).zfill(5)}.png"), dtype=np.uint8)
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
        video_frames = (frames.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).numpy()[0]   # B x T x C x H x W => T x H x W x C

        # final = []
        # for i in range(pixel_head.size(1)):
        #     img = video_frames[i]
        #     head = np.array(heads_pil[i])
        #     light = np.array(lights_drv_pil[i])
        #     tar = np.array(pixel_pil[i])
        #     ref = np.array(reference_pil[0])
        #     # final.append(np.concatenate([ref, head, light, img, tar], axis=1))
        #     Image.fromarray(np.uint8(np.concatenate([ref, light, img, tar], axis=1))).save(f"{save_path}/{str(i).zfill(5)}.png")

        # os.system(f"ffmpeg -r 20 -i {path_tmp}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")
        # torchvision.io.write_video(save_path, final, fps=20, video_codec='h264', options={'crf': '10'})
        
        res_frame = []
        shading_frame = []
        out_frame = []
        
        res_frame_256 = []
        shading_frame_256 = []
        out_frame_256 = []
        
        num_frames = video_frames.shape[0]
        save_path = f'{save_path}/gs={guidance}_ds={inference_steps}/n_frames={num_frames-1}'
        os.makedirs(save_path + '/256/', exist_ok=True)
        os.makedirs(save_path + '/512/', exist_ok=True)
        for i in range(num_frames):
            img = video_frames[i]
            light = np.array(lights_drv_pil[i])
            out = np.concatenate([img, light], axis=1)
            
            # 512x512
            out_frame.append(out)
            res_frame.append(img)
            shading_frame.append(light)
            
            Image.fromarray(np.uint8(img)).save(f"{save_path}/512/res_frame{str(i).zfill(3)}.png")
            Image.fromarray(np.uint8(light)).save(f"{save_path}/512/ren_frame{str(i).zfill(3)}.png")
            Image.fromarray(np.uint8(out)).save(f"{save_path}/512/out_frame{str(i).zfill(3)}.png")
            
            # 256x256
            img_256 = Image.fromarray(np.uint8(img)).resize((256, 256), Image.LANCZOS)
            light_256 = Image.fromarray(np.uint8(light)).resize((256, 256), Image.LANCZOS)
            out_256 = Image.fromarray(np.concatenate([np.array(img_256), np.array(light_256)], axis=1).astype(np.uint8))
            
            res_frame_256.append(np.array(img_256))
            shading_frame_256.append(np.array(light_256))
            out_frame_256.append(np.array(out_256))
            
            img_256.save(f"{save_path}/256/res_frame{str(i).zfill(3)}.png")
            light_256.save(f"{save_path}/256/ren_frame{str(i).zfill(3)}.png")
            out_256.save(f"{save_path}/256/out_frame{str(i).zfill(3)}.png")

        res_frame_rt = res_frame + res_frame[::-1]
        shading_frame_rt = shading_frame + shading_frame[::-1]
        out_frame_rt = out_frame + out_frame[::-1]
        res_frame_rt_256 = res_frame_256 + res_frame_256[::-1]
        shading_frame_rt_256 = shading_frame_256 + shading_frame_256[::-1]
        out_frame_rt_256 = out_frame_256 + out_frame_256[::-1]

        # frames to vids using ffmpeg and subprocess
        if save_vid:
            output_vid_name = ['out', 'ren', 'res']
            for reso in ['512', '256']:
                save_path_tmp = f'{save_path}/{reso}'
                os.makedirs(save_path_tmp, exist_ok=True)
                for i, fn in enumerate(['res_frame%03d.png', 'ren_frame%03d.png', 'out_frame%03d.png']):
                    input_pattern = f"{save_path_tmp}/{fn}"
                    output_path = f"{save_path_tmp}/{output_vid_name[i]}.mp4"
                    cmd = f"ffmpeg -r 24 -i {input_pattern} -pix_fmt yuv420p -c:v libx264 {output_path} -y"
                    # subprocess.run(cmd.split(' '), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)
                    os.system(cmd)

            # save the roundtrip version (forward + reverse)
            for reso in ['512', '256']:
                save_path_tmp = f'{save_path}/{reso}'
                vid_rt_list = [out_frame_rt, shading_frame_rt, res_frame_rt] if reso == '512' else [out_frame_rt_256, shading_frame_rt_256, res_frame_rt_256]
                for i, vf in enumerate(vid_rt_list):
                    torchvision.io.write_video(video_array=vf, filename=f"{save_path_tmp}/{output_vid_name[i]}_rt.mp4", fps=24, video_codec='h264', options={'crf': '17'})
        
        eval_dir = eval_dict["eval_dir"]
        if eval_dir is not None:
            src_id = eval_dict["src_id"]
            dst_id = eval_dict["dst_id"]
            for reso in ['256', '512']:
                eval_save_dir = f"{eval_dir}/out/{reso}/"
                os.makedirs(eval_save_dir, exist_ok=True)
                if reso == '256':
                    res_frame_ = res_frame_256
                else:
                    res_frame_ = res_frame
                f_relit = res_frame_[-1]
                os.makedirs(eval_save_dir, exist_ok=True)
                Image.fromarray(np.uint8(f_relit)).save(f"{eval_save_dir}/input={src_id}#pred={dst_id}.png")
                # torchvision.utils.save_image(tensor=f_relit, fp=f"{eval_save_dir}/input={src_id}#pred={dst_id}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pretrained_weights/stable-video-diffusion-img2vid-xt", help="stable video diffusion pretrained model path") 
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_weights/relipa", help="relightable portrait animation checkpoint path") 
    parser.add_argument("--guidance", type=float, default=4.5, help="lighting intensity")
    parser.add_argument("--inference_steps", type=int, default=25, help="diffusion reverse sampling steps")
    parser.add_argument("--driving_mode", type=str, default="relighting", help="relighting | landmark")
    
    # DiFaReli++'s running cfg for comparison
    parser.add_argument("--sample_pair_json", type=str, required=True, help="sample pair json file for DiFaReli++ comparison")
    parser.add_argument("--idx", nargs='+', type=int, default=[-1], help="index of the source spherical harmonics coefficients to rotate")
    parser.add_argument("--video_path", type=str, required=True, help="reference and shading") 
    parser.add_argument("--save_path", type=str, default="result.mp4", help="result save path")
    parser.add_argument("--scale_sh", type=float, required=True, help="scale spherical harmonics coefficients for DiFaReli++ comparison")
    parser.add_argument("--eval_dir", type=str, default=None, help="evaluation directory for DiFaReli++ comparison")
    parser.add_argument("--save_vid", action='store_true', default=False, help="save video or not")
    
    '''
    Save into
        - res_frame##.png
        - shading_frame##.png
        - video (out and shading)
        for out in [res_frame, shading_frame]:
            - out.mp4
            - out_nf.mp4    (No first frame - to match the DiFaReli++ (first is inversion))
            - out_rt.mp4    (Roundtrip version)
            - out_rt_nf.mp4 (No first frame + Roundtrip version)
        
    '''
    
    args = parser.parse_args()
    relightablepa = RelightablePA(pretrained_model_name_or_path=args.pretrained_model_name_or_path, checkpoint_path=args.checkpoint_path)
    
    with open(args.sample_pair_json, 'r') as f:
        sample_pairs = json.load(f)['pair']
        sample_pairs_k = [k for k in sample_pairs.keys()]
        sample_pairs_v = [v for v in sample_pairs.values()]
        
        
    if len(args.idx) > 2:
        # Filter idx to be within 0 < idx < len(sample_pairs)
        to_run_idx = [i for i in args.idx if 0 <= i < len(sample_pairs)]
    elif args.idx == [-1]:
        s = 0
        e = len(sample_pairs)
        to_run_idx = list(range(s, e))
    elif len(args.idx) == 2:
        s, e = args.idx
        s = max(0, s)
        e = min(e, len(sample_pairs))
        to_run_idx = list(range(s, e))
    else:
        raise ValueError("Invalid index range provided. Please provide a valid range or -1 for all indices.")

    logger = createLogger()
    
    logger.info("#" * 80)
    logger.warning("Relipa's running on...")
    logger.info(f"[#] Video path: {args.video_path}")
    logger.info(f"[#] Save path: {args.save_path}")
    logger.info(f"[#] Running idx: {to_run_idx}")
    logger.info(f"[#] Sample json: {args.sample_pair_json}")
    logger.info(f"[#] Scale SH: {args.scale_sh}")
    logger.info(f"[#] Evaluation dir: {args.eval_dir}")
    logger.info(f"[#] Save video: {args.save_vid}")
    logger.warning("Relipa's parameters...")
    logger.info(f"[#] Guidance: {args.guidance}")
    logger.info(f"[#] Inference steps: {args.inference_steps}")
    logger.info("#" * 80)
    
    to_run_idx = tqdm.tqdm(to_run_idx, desc="Processing indices", total=len(to_run_idx), unit="index")
    for idx in to_run_idx:
        pair = sample_pairs_v[idx]
        pair_id = sample_pairs_k[idx]
        fn = f'{pair_id}_src={pair["src"]}_dst={pair["dst"]}'
        
        to_run_idx.set_description(f"[#] Processing index {idx} with src image {pair['src']} and dst image {pair['dst']}...")
        
        input_path = f'{args.video_path}/{fn}/'
        video_path = f'{args.video_path}/{fn}/{fn}.mp4'
        if not os.path.exists(video_path):
            logger.error(f"[!] Input video {input_path} does not exist. Skipping index {idx}.")
            continue
        
        save_path = f'{args.save_path}/src={pair["src"]}/dst={pair["dst"]}/scale_sh={args.scale_sh}/'
        relightablepa.portrait_animation_and_relighting(video_path=input_path, 
                                                        save_path=save_path, 
                                                        guidance=args.guidance, 
                                                        inference_steps=args.inference_steps, 
                                                        driving_mode=args.driving_mode,
                                                        save_vid=args.save_vid,
                                                        eval_dict={"src_id": pair["src"], "dst_id": pair["dst"], "eval_dir": args.eval_dir})
