
import json
import random
from typing import List

import torch
import torch.nn.functional as F 
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 


class TalkingHeadVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        data_meta_path="data.json",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.vid_meta = json.load(open(data_meta_path, "r"))

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        light_path = video_meta["light_path"]
        mask_path = video_meta["mask_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        light_reader = VideoReader(light_path)
        mask_reader = VideoReader(mask_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read ref frame, kps and mesh
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())
        ref_pose = Image.fromarray(kps_reader[ref_img_idx].asnumpy())
        ref_light = Image.fromarray(light_reader[ref_img_idx].asnumpy())
        ref_mask = Image.fromarray(mask_reader[ref_img_idx].asnumpy())

        # read frames, kps and meshes
        vid_pil_image_list = [ref_img]
        pose_pil_image_list = [ref_pose]
        light_pil_image_list = [ref_light]
        mask_pil_image_list = [ref_mask]
        for index in batch_index[1:]:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = light_reader[index]
            light_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = mask_reader[index]
            mask_pil_image_list.append(Image.fromarray(img.asnumpy()))

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_light = self.augmentation(
            light_pil_image_list, self.cond_transform, state
        )
        pixel_values_mask = self.augmentation(
            mask_pil_image_list, self.cond_transform, state
        )
        pixel_values_mask[pixel_values_mask > 0.5] = 1.0 
        pixel_values_mask[pixel_values_mask <= 0.5] = 0.0
        pixel_values_mask = F.interpolate(pixel_values_mask, [self.height//8, self.width//8])[:, 0:1]

        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        pixel_values_ref_mask = self.augmentation(ref_mask, self.cond_transform, state)
        pixel_values_ref_mask[pixel_values_ref_mask > 0.5] = 1.0 
        pixel_values_ref_mask[pixel_values_ref_mask <= 0.5] = 0.0
        pixel_values_ref_mask = pixel_values_ref_mask.unsqueeze(0)
        pixel_values_ref_mask = F.interpolate(pixel_values_ref_mask, [self.height//8, self.width//8])[:, 0:1]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_head=pixel_values_pose,
            pixel_values_light=pixel_values_light,
            pixel_values_mask=pixel_values_mask,
            pixel_values_ref_mask=pixel_values_ref_mask,
            pixel_values_ref_img=pixel_values_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)



# import torch 
# import time 
# from PIL import Image 

# train_dataset = TalkingHeadVideoDataset(sample_rate=4, n_sample_frames=16, width=512, height=512, data_meta_paths=["/media/Data/gmt/Dataset/TalkingHeadVideo/VFHQ/VFHQ-data.json"])
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     shuffle=True,
#     batch_size=4,
#     num_workers=2,
#     pin_memory=False
# )

# from tqdm import tqdm 
# for idx, batch in tqdm(enumerate(train_dataloader)):

#     img = batch["pixel_values_vid"].permute(0, 1, 3, 4, 2).numpy()
#     head = batch["pixel_values_head"].permute(0, 1, 3, 4, 2).numpy()
#     light = batch["pixel_values_light"].permute(0, 1, 3, 4, 2).numpy()
#     nolight = batch["pixel_values_mask"].permute(0, 1, 3, 4, 2).numpy()
#     ref_img = batch["pixel_values_ref_img"].repeat(1, 16, 1, 1, 1).permute(0, 1, 3, 4, 2).numpy()
#     ref_nolight = batch["pixel_values_ref_mask"].repeat(1, 16, 1, 1, 1).permute(0, 1, 3, 4, 2).numpy()
#     # print(audio_emb)
#     # print(img.shape, tgt_pose.shape, ref_img.shape, face_mask.shape, "111")
#     # ref_pose = batch["pixel_values_ref_pose"]#.permute(0, 2, 3, 1).numpy()
#     img = (img + 1) * 127.5
#     head = head * 255
#     light = light * 255
#     nolight = nolight * 255
#     ref_img = (ref_img + 1) * 127.5
#     ref_nolight = ref_nolight * 255

#     imgs = []
#     heads = []
#     lights = []
#     nolights = []
#     refs = []
#     refs_nolight = []
#     for i in range(16):
#         imgs.append(img[0, i])
#         heads.append(head[0, i])
#         lights.append(light[0, i])
#         nolights.append(nolight[0, i])
#         refs.append(ref_img[0, i])
#         refs_nolight.append(ref_nolight[0, i])
#     imgs = np.concatenate(imgs, axis=1)
#     heads = np.concatenate(heads, axis=1)
#     lights = np.concatenate(lights, axis=1)
#     nolights = np.concatenate(nolights, axis=1)
#     refs = np.concatenate(refs, axis=1)
#     refs_nolight = np.concatenate(refs_nolight, axis=1)
#     fuse_head = imgs * 0.5 + heads * 0.5 
#     fuse_light = imgs * 0.5 + lights * 0.5
#     print(imgs.shape, lights.shape, heads.shape, fuse_head.shape)
#     con = np.concatenate([imgs, heads, lights, nolights, lights-nolights, fuse_head, fuse_light, refs, refs_nolight], axis=0)
#     Image.fromarray(np.uint8(con)).save("con.jpg")
#     break 