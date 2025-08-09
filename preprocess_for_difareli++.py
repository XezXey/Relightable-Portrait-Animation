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
from utils.sh_utils import rotate_sh

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
    def __init__(self, light_path) -> None:
        self.vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
        self.lmk_extractor = LMKExtractor()

        self.fm = FaceMatting()

        self.fir = FaceImageRender()

        self.fkpd = FaceKPDetector()
        
        self.light_path = light_path
        
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
        

    def inference(self, source_path, target_light_name, num_frames, save_path):
        tmp_path = "resources_difareli++/target/"

        if os.path.exists(tmp_path):
            os.system(f"rm -r {tmp_path}")
            
        os.mkdir(tmp_path)

        # motion sync
        source_image_name = os.path.basename(source_path)
        source_image = np.array(Image.open(source_path).resize([512, 512]))[..., :3]
        if target_light_name:
            target_light = self.read_sh[target_light_name.replace('.png', '.jpg')]        
            target_light = torch.tensor(target_light).cuda()
        else:
            raise ValueError("[#] Please specify the target light name that exists in the light_path txt file.")
        
        # Self-drive for DiFaReli++ comparison (Single image relighting)
        self_driver_frames = [source_image.copy() for _ in range(num_frames)]
        
        aligned_kpmaps = self.fkpd.motion_self(self_driver_frames)
        
        alpha = self.fm.portrait_matting(source_image)
        
        target_light = rotate_sh({'light': target_light.detach().cpu().numpy()}, src_idx=0, n_step=num_frames, axis=2)  # Rotate the light for each frame
        target_light = target_light['light'].reshape(num_frames, 9, 3)  # Reshape to [num_frames, 9, 3]
        target_light = torch.tensor(target_light).cuda()  # Convert to tensor and move to GPU
        
        # Input light to render need to be [1, 9, 3]
        # aligned_shading = self.fir.render_with_given_light_difarelipp(source_image, torch.tensor(target_light.reshape(1, 9, 3)).cuda().repeat_interleave(dim=0, repeats=10))
        aligned_shading = self.fir.render_with_given_light_difarelipp(source_image, target_light)
        
        for idx, (drv_frame, kpmap, shading) in tqdm(enumerate(zip(self_driver_frames, aligned_kpmaps, aligned_shading))):
            img = np.concatenate([source_image, alpha, drv_frame, kpmap, shading], axis=1)
            Image.fromarray(np.uint8(img)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")

        source_kp = self.fkpd.single_kp(source_image)
        source_shading = self.fir.render_motion_single_with_light(source_image, source_image)
        
        img = np.concatenate([source_image, alpha, source_image, source_kp, source_shading], axis=1)
        Image.fromarray(np.uint8(img)).save(f"{tmp_path}/{str(0).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="input image path") 
    parser.add_argument("--light_path", type=str, required=True, help="estimated lighting path (.txt)") 
    parser.add_argument("--save_path", type=str, default="resources/shading.mp4", help="shading hints") 
    parser.add_argument("--num_frames", type=int, default=30, help="number of frames to generate for self-drive")
    parser.add_argument("--sample_pair_json", type=str, required=True, help="sample pair json file for DiFaReli++ comparison")
    parser.add_argument("--idx", nargs='+', type=int, default=[-1], help="index of the source spherical harmonics coefficients to rotate")
    args = parser.parse_args()


    with open(args.sample_pair_json, 'r') as f:
        sample_pairs = json.load(f)['pair']
        sample_pairs = [v for v in sample_pairs.values()]
        
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


    iv = InferImage(light_path=args.light_path)
    for idx in to_run_idx:
        print(f"[#] Processing index {idx} with source image {sample_pairs[idx]['src']} and target light {sample_pairs[idx]['dst']}...")
        
        source_img = sample_pairs[idx]['src'].replace('.jpg', '.png')
        target_light_name = sample_pairs[idx]['dst'].replace('.jpg', '.png')
        iv.inference(source_path=args.source_path + source_img, 
                    save_path=args.save_path, 
                    target_light_name=target_light_name, 
                    num_frames=args.num_frames
                    )
            
            