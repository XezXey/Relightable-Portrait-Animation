from src.facepose.mp_utils  import LMKExtractor
from src.facepose.draw_utils import FaceMeshVisualizer
from src.facepose.motion_utils import motion_sync
from src.facematting.u2net_matting import U2NET
from src.decalib.utils import util
from src.decalib.utils.tensor_cropper import transform_points
from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import torch
import cv2 
import os 
import argparse 
from utils.sh_utils import rotate_sh, interp_sh
from utils import logging
import subprocess

# ignore warnings
import warnings
warnings.filterwarnings('ignore')



class FaceMatting:
    def __init__(self) -> None:
        self.net = U2NET(3,1).cuda()
        self.net.load_state_dict(torch.load("./src/facematting/u2net_human_seg.pth"))

    def portrait_matting(self, rgb_image):
        rgb_image = cv2.resize(rgb_image, (320, 320))[None] / 255
        rgb_image[:,:,0] = (rgb_image[:,:,0] - 0.485) / 0.229
        rgb_image[:,:,1] = (rgb_image[:,:,1] - 0.456) / 0.224
        rgb_image[:,:,2] = (rgb_image[:,:,2] - 0.406) / 0.225
        rgb_image_th = torch.tensor(rgb_image, dtype=torch.float32).cuda().permute(0, 3, 1, 2)
        with torch.no_grad():
            d1,d2,d3,d4,d5,d6,d7 = self.net(rgb_image_th)
            # normalization
            pred = d1[:,0,:,:]
            ma = torch.max(pred)
            mi = torch.min(pred)
            alpha = (pred-mi)/(ma-mi)
        alpha = alpha.detach().cpu().numpy()[0]
        alpha[alpha > 0.5] = 255 
        alpha[alpha <=0.5] = 0
        alpha = np.dstack([alpha, alpha, alpha])
        alpha = cv2.resize(alpha, (512, 512))
        alpha = cv2.dilate(alpha, np.ones([7, 7]))
        return alpha


class FaceImageRender:
    def __init__(self) -> None:
        # Init DECA
        self.deca = DECA(config=deca_cfg)
        f_mask = np.load('./src/decalib/data/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
        v_mask = np.load('./src/decalib/data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
        self.mask={
            'v_mask':v_mask['face'].tolist(),
            'f_mask':f_mask['face'].tolist()
        }
    
    def image_to_3dcoeff(self, rgb_image):
        with torch.no_grad():
            codedict, detected_flag = self.deca.img_to_3dcoeff(rgb_image)
        return codedict

    def render_shape(self, shape, exp, pose, cam, light, tform, h, w):
        with torch.no_grad():
            # all parameters are from codedict
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)

            ## projection
            trans_verts = util.batch_orth_proj(verts, cam); trans_verts[:,:,1:] = -trans_verts[:,:,1:]

            points_scale = [self.deca.image_size, self.deca.image_size]
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])

            shape_images, _, grid, alpha_images, albedo_images =self.deca.render.render_shape(verts, trans_verts, h=h, w=w, lights=light, images=None, return_grid=True, mask=self.mask)
            shape_images = shape_images.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
            albedo_images = albedo_images.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255
            # shape_images = shape_images.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy()[0] * 255
            # albedo_images = albedo_images.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy()[0] * 255
        return shape_images, albedo_images

    def render_shape_with_light(self, codedict, target_light=None):
        if target_light is None:
            target_light = codedict["light"]
        shape, exp, pose = codedict["shape"], codedict["exp"], codedict["pose"]
        cam, tform, h, w = codedict["cam"], codedict["tform"], codedict["height"], codedict["width"]
        shape_image, albedo_image = self.render_shape(shape, exp, pose, cam, target_light, tform, h, w)
        return shape_image
    
    def render_with_given_light_difarelipp(self, image, target_light):
        codedict = self.image_to_3dcoeff(image) 
        B = target_light.shape[0]
        # Expand all codedict values to match the batch size of target_light
        for key in codedict:
            if isinstance(codedict[key], torch.Tensor):
                codedict[key] = codedict[key].clone().detach().repeat_interleave(dim=0, repeats=B)
        shading = self.render_shape_with_light(codedict, target_light=target_light)
        return shading 

    def render_motion_single(self, image):
        codedict = self.image_to_3dcoeff(image) 
        shading = self.render_shape_with_light(codedict)
        return shading 

    def render_motion_single_with_light(self, image, target_light_image):
        codedict = self.image_to_3dcoeff(image) 
        target_light = self.image_to_3dcoeff(target_light_image)["light"]
        shading = self.render_shape_with_light(codedict, target_light=target_light)
        return shading 
    
    def render_motion_sync(self, ref_image, driver_frames, target_light_image):
        ref_code_dict = self.image_to_3dcoeff(ref_image)
        target_light = self.image_to_3dcoeff(target_light_image)["light"]

        shading_frames = []
        for drv_frm in tqdm(driver_frames):
            codedict = self.image_to_3dcoeff(drv_frm)
            shape, exp, pose = ref_code_dict["shape"], ref_code_dict["exp"], codedict["pose"]
            cam, tform, h, w = ref_code_dict["cam"], ref_code_dict["tform"], ref_code_dict["height"], ref_code_dict["width"]
            shape_image, albedo_image = self.render_shape(shape, exp, pose, cam, target_light, tform, h, w)
            shading_frames.append(shape_image)
        return shading_frames

    def render_motion_sync_relative(self, ref_image, driver_frames, target_light_image):
        ref_codedict = self.image_to_3dcoeff(ref_image)
        target_light = self.image_to_3dcoeff(target_light_image)["light"]

        drv_codedict_list = []
        shading_frames = []
        for drv_frm in tqdm(driver_frames):
            drv_codedict = self.image_to_3dcoeff(drv_frm)
            drv_codedict_list.append(drv_codedict)
        
        # best_dist = 10000
        # best_pose = None 
        # for idx, drv_codedict in enumerate(drv_codedict_list):
        #     dist = torch.mean(torch.abs(ref_codedict["pose"] - drv_codedict["pose"]))
        #     if dist < best_dist:
        #         best_dist = dist
        #         best_pose = drv_codedict["pose"]
        best_pose = drv_codedict_list[0]["pose"]
        best_exp = drv_codedict_list[0]["exp"]
        for drv_codedict in drv_codedict_list:
            relative_pose = drv_codedict["pose"] - best_pose + ref_codedict["pose"]
            relative_exp = drv_codedict["exp"] - best_exp + ref_codedict["exp"]
            shape, exp, pose = ref_codedict["shape"], relative_exp, relative_pose
            cam, tform, h, w = ref_codedict["cam"], ref_codedict["tform"], ref_codedict["height"], ref_codedict["width"]
            shape_image, albedo_image = self.render_shape(shape, exp, pose, cam, target_light, tform, h, w)
            shading_frames.append(shape_image)
        return shading_frames

    def render_motion_sync(self, ref_image, driver_frames, target_light_image):
        ref_codedict = self.image_to_3dcoeff(ref_image)
        target_light = self.image_to_3dcoeff(target_light_image)["light"]

        drv_codedict_list = []
        shading_frames = []
        for drv_frm in tqdm(driver_frames):
            drv_codedict = self.image_to_3dcoeff(drv_frm)
            drv_codedict_list.append(drv_codedict)
        
        for drv_codedict in drv_codedict_list:
            shape, exp, pose = ref_codedict["shape"], drv_codedict["exp"], drv_codedict["pose"]
            cam, tform, h, w = ref_codedict["cam"], ref_codedict["tform"], ref_codedict["height"], ref_codedict["width"]
            shape_image, albedo_image = self.render_shape(shape, exp, pose, cam, target_light, tform, h, w)
            shading_frames.append(shape_image)
        return shading_frames

class FaceKPDetector:
    def __init__(self) -> None:
        self.vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
        self.lmk_extractor = LMKExtractor()

    def motion_sync(self, ref_image, driver_frames):        
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
        ref_frame =cv2.resize(ref_image, (512, 512))
        ref_det = self.lmk_extractor(ref_frame)

        sequence_driver_det = []
        try: 
            for frame in tqdm(driver_frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame =cv2.resize(frame, (512, 512))
                result = self.lmk_extractor(frame)
                assert result is not None, "bad video, face not detected"
                sequence_driver_det.append(result)
        except:
            print("face detection failed")
            exit()

        sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
        pose_frames = [self.vis.draw_landmarks((512, 512), i, normed=False) for i in sequence_det_ms]
        return pose_frames

    def motion_self(self, driver_frames):        
        pose_frames = []
        try: 
            for frame in tqdm(driver_frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame =cv2.resize(frame, (512, 512))
                frame_det = self.lmk_extractor(frame)
                kpmap = self.vis.draw_landmarks((512, 512), frame_det["lmks"], normed=True)
                pose_frames.append(kpmap)
        except:
            print("face detection failed")
            exit()

        return pose_frames

    def single_kp(self, image):
        frame_det = self.lmk_extractor(image)
        kpmap = self.vis.draw_landmarks((512, 512), frame_det["lmks"], normed=True)
        return kpmap 

class InferVideo:
    def __init__(self) -> None:
        self.vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
        self.lmk_extractor = LMKExtractor()

        self.fm = FaceMatting()

        self.fir = FaceImageRender()

        self.fkpd = FaceKPDetector()

    def inference(self, source_path, light_path, video_path, save_path, motion_align="relative"):
        tmp_path = "resources/target/"

        if os.path.exists(tmp_path):
            os.system(f"rm -r {tmp_path}")
            
        os.mkdir(tmp_path)
        os.system(f"ffmpeg -i {video_path} {tmp_path}/%5d.png")

        # motion sync
        source_image = np.array(Image.open(source_path).resize([512, 512]))[..., :3]
        target_lighting = np.array(Image.open(light_path).resize([512, 512]))[..., :3]
        
        driver_frames = [np.array(Image.open(os.path.join(tmp_path, str(i).zfill(5)+".png")).resize([512, 512])) for i in range(1, 1 + len(os.listdir(tmp_path)))]
        
        aligned_kpmaps = self.fkpd.motion_self(driver_frames)
        
        alpha = self.fm.portrait_matting(source_image)

        if motion_align == "relative":
            aligned_shading = self.fir.render_motion_sync_relative(source_image, driver_frames, target_lighting)
        else:
            aligned_shading = self.fir.render_motion_sync(source_image, driver_frames, target_lighting)
        
        
        for idx, (drv_frame, kpmap, shading) in tqdm(enumerate(zip(driver_frames, aligned_kpmaps, aligned_shading))):
            img = np.concatenate([source_image, alpha, drv_frame, kpmap, shading], axis=1)
            Image.fromarray(np.uint8(img)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")

        source_kp = self.fkpd.single_kp(source_image)
        source_shading = self.fir.render_motion_single_with_light(source_image, source_image)
        
        img = np.concatenate([source_image, alpha, source_image, source_kp, source_shading], axis=1)
        Image.fromarray(np.uint8(img)).save(f"{tmp_path}/{str(0).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")

class InferImage:
    def __init__(self, mani_light_dict) -> None:
        self.vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
        self.lmk_extractor = LMKExtractor()

        self.fm = FaceMatting()

        self.fir = FaceImageRender()

        self.fkpd = FaceKPDetector()
        
        self.light_path = mani_light_dict['light_path']
        self.mani_light = mani_light_dict['mani_light']
        self.rotate_sh_axis = mani_light_dict['rotate_sh_axis']
        self.scale_sh = mani_light_dict['scale_sh']
        
        if args.light_path.endswith(".txt"):
            self.read_sh = self.read_sh_from_txt()
        else: 
            raise NotImplementedError("[#] Only .txt format is supported for lighting path.")

    def read_sh_from_txt(self):
        light = pd.read_csv(self.light_path, header=None, sep=" ", index_col=False, lineterminator='\n')
        light.rename(columns={0:'img_name'}, inplace=True)
        light = light.set_index('img_name').T.to_dict('list')
        for k, v in light.items():
            light[k] = np.array(v, dtype=np.float64)[None, ...] # [1, 27]
        return light 
        

    def inference(self, source_path, target_light_name, num_frames, save_path, save_fn):
        os.makedirs(save_path, exist_ok=True)

        # motion sync
        source_image_name = os.path.basename(source_path)
        source_image = np.array(Image.open(source_path).resize([512, 512]))[..., :3]
        if target_light_name:
            if args.img_ext == '.png':
                # sample_json has .jpg but the image is .png
                source_light = self.read_sh[source_image_name.replace('.png', '.jpg')]
                target_light = self.read_sh[target_light_name.replace('.png', '.jpg')]
            else:
                source_light = self.read_sh[source_image_name]
                target_light = self.read_sh[target_light_name]
            source_light = torch.tensor(source_light).cuda()
            target_light = torch.tensor(target_light).cuda()
        else:
            raise ValueError("[#] Please specify the target light name that exists in the light_path txt file.")
        
        # Self-drive for DiFaReli++ comparison (Single image relighting)
        source_kp = self.fkpd.single_kp(source_image)
        self_driver_frames = [source_image.copy() for _ in range(num_frames)]
        all_kpmaps = [source_kp.copy() for _ in range(num_frames)]
        
        alpha = self.fm.portrait_matting(source_image)
        
        if self.mani_light == "rotate_sh":
            target_light = rotate_sh({'light': target_light.detach().cpu().numpy()}, src_idx=0, n_step=num_frames, axis=self.rotate_sh_axis)  # Rotate the light for each frame
        elif self.mani_light == 'interp_sh':
            target_light = interp_sh({'source_light': source_light.detach().cpu().numpy(), 'target_light': target_light.detach().cpu().numpy()}, n_step=num_frames)  # Rotate the light for each frame
        else:
            raise NotImplementedError("[#] Only 'rotate_sh' and 'interp_sh' are supported for manipulated light.")

        target_light = target_light['light'].reshape(num_frames, 9, 3)  # Reshape to [num_frames, 9, 3]
        target_light = torch.tensor(target_light).cuda()  # Convert to tensor and move to GPU
        target_light = target_light * args.scale_sh
        
        # Input light to render need to be [1, 9, 3]
        # aligned_shading = self.fir.render_with_given_light_difarelipp(source_image, torch.tensor(target_light.reshape(1, 9, 3)).cuda().repeat_interleave(dim=0, repeats=10))
        aligned_shading = self.fir.render_with_given_light_difarelipp(source_image, target_light)
        
        img_path = f"{save_path}/{save_fn}/"
        os.makedirs(img_path, exist_ok=True)
        # t = [1, num_frames]
        for idx, (drv_frame, kpmap, shading) in tqdm(enumerate(zip(self_driver_frames, all_kpmaps, aligned_shading)), desc='[#] Saving frames...', leave=False):
            img = np.concatenate([source_image, alpha, drv_frame, kpmap, shading], axis=1)
            Image.fromarray(np.uint8(img)).save(f"{img_path}/{str(idx + 1).zfill(5)}.png")

        # t = [0]
        source_shading = self.fir.render_motion_single_with_light(source_image, source_image)[0]
        img = np.concatenate([source_image, alpha, source_image, source_kp, source_shading], axis=1)
        Image.fromarray(np.uint8(img)).save(f"{img_path}/{str(0).zfill(5)}.png")
        
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(
                [
                    "ffmpeg",
                    "-r", "24",
                    "-i", f"{img_path}/%05d.png",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    f'{img_path}/{save_fn}.mp4',
                    "-y"
                ],
                stdout=devnull,
                stderr=devnull
            )
        
        vid_path = f'{save_path}/vids/'
        os.makedirs(vid_path, exist_ok=True)
        # os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(
                [
                    "ffmpeg",
                    "-r", "24",
                    "-i", f"{img_path}/%05d.png",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    f'{vid_path}/{save_fn}.mp4',
                    "-y"
                ],
                stdout=devnull,
                stderr=devnull
            )
            

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="input image path") 
    parser.add_argument("--light_path", type=str, required=True, help="estimated lighting path (.txt)") 
    parser.add_argument("--save_path", type=str, default="resources/shading.mp4", help="shading hints") 
    parser.add_argument("--num_frames", type=int, default=30, help="number of frames to generate for self-drive")
    parser.add_argument("--sample_pair_json", type=str, required=True, help="sample pair json file for DiFaReli++ comparison")
    parser.add_argument("--idx", nargs='+', type=int, default=[-1], help="index of the source spherical harmonics coefficients to rotate")
    parser.add_argument("--use_self_light", action="store_true", help="use self-lighting for rendering")
    parser.add_argument("--mani_light", type=str, required=True, help="manipulated light path for DiFaReli++ comparison")
    parser.add_argument("--rotate_sh_axis", type=int, default=2, help="axis to rotate spherical harmonics coefficients, 0 for x, 1 for y, 2 for z")
    parser.add_argument("--scale_sh", type=float, default=1.0, help="scale factor for spherical harmonics coefficients")
    parser.add_argument("--img_ext", type=str, required=True, help="image extension in the sample pair json file, e.g., .jpg or .png")
    args = parser.parse_args()


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


    
    logger = logging.createLogger()
    
    logger.info("#" * 80)
    logger.info(f"[#] Light manipulation mode: {args.mani_light}")
    logger.info(f"[#] Light rotation axis (affective if mode is rotate_sh): {args.rotate_sh_axis}")
    logger.info(f"[#] Scale sh: {args.scale_sh}")
    logger.info(f"[#] Use self light: {args.use_self_light}")
    logger.info(f"[#] Number of frames to generate for self-drive: {args.num_frames}")
    logger.info(f"[#] Source image path: {args.source_path}")
    logger.info(f"[#] Save path: {args.save_path}")
    logger.info("#" * 80)

    mani_light_dict = {
        "light_path": args.light_path,
        "mani_light": args.mani_light,
        "rotate_sh_axis": args.rotate_sh_axis,
        "scale_sh": args.scale_sh
    }
    iv = InferImage(mani_light_dict=mani_light_dict)

    to_run_idx = tqdm(to_run_idx, desc="Processing indices", total=len(to_run_idx), unit="index")
    for idx in to_run_idx:
        pair = sample_pairs_v[idx]
        pair_id = sample_pairs_k[idx]
        
        to_run_idx.set_description(f"[#] Processing index {idx} with src image {pair['src']} and dst image {pair['dst']}...")
        
        if args.img_ext == '.png':
            source_img = pair['src'].replace('.jpg', '.png')
            if args.use_self_light:
                target_light_name = pair['src'].replace('.jpg', '.png')
            else:
                target_light_name = pair['dst'].replace('.jpg', '.png')
        else:
            source_img = pair['src']
            if args.use_self_light:
                target_light_name = pair['src']
            else:
                target_light_name = pair['dst']
        
        mode_suffix = f'/{args.mani_light}/' if args.mani_light == 'interp_sh' else f'/{args.mani_light}_axis={args.rotate_sh_axis}/'
        save_path = f'{args.save_path}/{mode_suffix}/scale_sh={args.scale_sh}/n_step={args.num_frames}/'
        save_fn = f'{pair_id}_src={pair["src"]}_dst={pair["dst"]}'
        iv.inference(source_path=args.source_path + source_img, 
                    save_path=save_path, 
                    save_fn=save_fn,
                    target_light_name=target_light_name, 
                    num_frames=args.num_frames
                    )
            
            