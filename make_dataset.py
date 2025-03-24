import os 
import json 
import numpy as np 
import argparse
from tqdm import tqdm 
from PIL import Image 
from preprocess import InferVideo 


class MakeDataset(InferVideo):
    def make_dataset(self, video_path, video_name, save_path, dataset="VFHQ"):
        tmp_path = "resources/target/"

        if os.path.exists(tmp_path):
            os.system(f"rm -r {tmp_path}")
            
        os.mkdir(tmp_path)
        os.system(f"ffmpeg -i {video_path}/{video_name} {tmp_path}/%5d.png")

        frames = [np.array(Image.open(os.path.join(tmp_path, str(i).zfill(5)+".png")).resize([512, 512])) for i in range(1, 1 + len(os.listdir(tmp_path)))]   
        kpmaps = self.fkpd.motion_self(frames)
        alphas = [self.fm.portrait_matting(frame) for frame in frames]
        shadings = [self.fir.render_motion_single(frame) for frame in frames]
        
        if not os.path.exists(f"{save_path}/{dataset}/{dataset}-video"):
            os.mkdir(f"{save_path}/{dataset}/{dataset}-video")
        if not os.path.exists(f"{save_path}/{dataset}/{dataset}-kpmap"):
            os.mkdir(f"{save_path}/{dataset}/{dataset}-kpmap")
        if not os.path.exists(f"{save_path}/{dataset}/{dataset}-mask"):
            os.mkdir(f"{save_path}/{dataset}/{dataset}-mask")
        if not os.path.exists(f"{save_path}/{dataset}/{dataset}-mesh"):
            os.mkdir(f"{save_path}/{dataset}/{dataset}-mesh")

        for idx, frame in enumerate(frames):
            Image.fromarray(np.uint8(frame)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path}/{dataset}/{dataset}-video/{video_name} -y")

        for idx, kpmap in enumerate(kpmaps):
            Image.fromarray(np.uint8(kpmap)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path}/{dataset}/{dataset}-kpmap/{video_name} -y")

        for idx, alpha in enumerate(alphas):
            Image.fromarray(np.uint8(alpha)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path}/{dataset}/{dataset}-mask/{video_name} -y")

        for idx, shading in enumerate(shadings):
            Image.fromarray(np.uint8(shading)).save(f"{tmp_path}/{str(idx + 1).zfill(5)}.png")
        os.system(f"ffmpeg -r 20 -i {tmp_path}/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path}/{dataset}/{dataset}-mesh/{video_name} -y")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./VFHQ", help="VFHQ | CelebV-HQ path")
    parser.add_argument("--save_path", type=str, default="./TalkingHeadVideo", help="save path")
    parser.add_argument("--dataset", type=str, default="VFHQ", help="save path")

    args = parser.parse_args()

    md = MakeDataset() 

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(os.path.join(args.save_path, args.dataset)):
        os.mkdir(os.path.join(args.save_path, args.dataset))

    data = []
    video_files = os.listdir(args.video_path)
    for video_file in tqdm(video_files):
        try:
            item = {}
            md.make_dataset(args.video_path, video_file, args.save_path, dataset=args.dataset)
            item["video_path"] = f"{args.save_path}/{args.dataset}/{args.dataset}-mesh/{video_file}"
            item["kps_path"] = f"{args.save_path}/{args.dataset}/{args.dataset}-kpmap/{video_file}"
            item["light_path"] = f"{args.save_path}/{args.dataset}/{args.dataset}-mesh/{video_file}"
            item["mask_path"] = f"{args.save_path}/{args.dataset}/{args.dataset}-mask/{video_file}"
            data.append(item)
        except:
            print("Error Video: ", video_file)
            continue
    
    filename = f"{args.save_path}/{args.dataset}/{args.dataset}-data-consistent.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        

