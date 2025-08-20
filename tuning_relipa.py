import argparse
import subprocess
import shlex
import os
import re
from utils.logging import createLogger
logger = createLogger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guidance",
        nargs="+",
        required=True,
        help="List of guidance values, e.g. --guidance 3.5 4.0 4.5"
    )
    parser.add_argument(
        "--video_path",
        nargs="+",
        required=True,
        help="List of video paths, e.g. --video_path /path1 /path2"
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Base save path"
    )
    parser.add_argument(
        "--sample_pair_json",
        required=True,
        help="Path to sample_pair_json file"
    )
    
    parser.add_argument(
        "--idx",
        nargs='+',
        required=True,
        help="index to run"
    )
    parser.add_argument(
        "--gpu_id",
        default="0",
        help="GPU ID to use"
    )
    args = parser.parse_args()

    # Fixed part of the command
    base_cmd = [
        "python", "inference_for_difareli++.py",
        "--pretrained_model_name_or_path", "/data2/mint/pretrained_weights/stable-video-diffusion-img2vid-xt",
        "--checkpoint_path", "/data2/mint/pretrained_weights/relipa",
        "--inference_steps", "25",
        "--driving_mode", "relighting",
    ]

    # Iterate over all combinations
    for g in args.guidance:
        for vp in args.video_path:
            # Construct unique save path for each run
            if 'rotate_sh_axis=1' in vp:
                mani_light = 'rotate_sh'
                rotate_sh_axis = '1'
            elif 'rotate_sh_axis=2' in vp:
                mani_light = 'rotate_sh'
                rotate_sh_axis = '2'
            elif 'interp_sh' in vp:
                mani_light = 'interp_sh'
                rotate_sh_axis = None
            else: 
                raise ValueError(f"[#] Unknown manipulation light type in video path: {vp}")
            
            match = re.search(r"scale_sh=([0-9.]+)", vp)
            if match:
                scale_sh = match.group(1)
        
            if mani_light == 'interp_sh':
                save_dir = f'{args.save_path}/{mani_light}/'
            elif mani_light == 'rotate_sh':
                save_dir = f'{args.save_path}/rotate_sh_axis={rotate_sh_axis}/'


            logger.warning("Relipa's running on...")
            logger.info(f"[#] Index: {args.idx}")
            logger.info(f"[#] Video path: {vp}")
            logger.info(f"[#] Save path: {save_dir}")
            logger.info(f"[#] Guidance: {g}")
            logger.info(f"[#] Manipulation light: {mani_light}")
            logger.info(f"[#] Rotate SH axis: {rotate_sh_axis}")
            logger.info(f"[#] Scale SH: {scale_sh}")
            logger.info(f"[#] GPU ID: {args.gpu_id}")
            
            cmd = (
                f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
                + " ".join(base_cmd)
                + f" --guidance {g}"
                + f" --video_path {shlex.quote(vp)}"
                + f" --save_path {shlex.quote(save_dir)}"
                + f" --sample_pair_json {shlex.quote(args.sample_pair_json)}"
                + f" --scale_sh {scale_sh}"
                + f" --idx {' '.join(args.idx)}"
            )

            logger.warning(f"Running command: {cmd}")
            subprocess.run(cmd, shell=False, check=True)

if __name__ == "__main__":
    main()
