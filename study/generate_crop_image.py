import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
from lib.utils.demo_utils import video_to_images
import torch
import numpy as np
import glob
from multi_person_tracker import MPT
import os
import os.path as osp
from lib.data_utils.img_utils import get_single_image_crop_demo

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def find_frame_idx(f, frames):
    exact_frame = np.where(frames >= f)
    if len(exact_frame[0]) > 0:
        return exact_frame[0][0]
    else:
        return np.where(frames < f)[0][-1]

def load_video_crop(folder_angle, subject_id, condition, walk_id, angle):
    image_folder = '/home/epinyoan/dataset/casia-b/dataset_b/all/images/'+subject_id+'-'+condition+'-'+walk_id+'-'+angle
    s_folder = '/home/epinyoan/dataset/casia-b/dataset_b/all/silhouettes/'+subject_id+'/'+condition+'-'+walk_id+'/'+angle #+'/'+subject_id+'-'+condition+'-'+walk_id+'-'+angle+'-'
    mot = MPT(
        device=device,
        batch_size=12,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < 25:
            del tracking_results[person_id]
            
    if len(list(tracking_results.keys())) == 0:
        with open("not_detect.txt", "a") as f:
            f.write(image_folder+"\n")
        return
    person_id = list(tracking_results.keys())[0]
    bboxes = tracking_results[person_id]['bbox']
    frames = tracking_results[person_id]['frames']



    s_img_paths = glob.glob(s_folder+'/*')
    silhuette_imgs = []
    norm_imgs = []
    final_bboxes = []
    final_frames = []
    for i, s_img_path in enumerate(sorted(s_img_paths)):
        # 1. video
        s_img = cv2.imread(s_img_path)
        f = int(s_img_path.split('/')[-1].split('.')[0].split('-')[-1])
        idx = find_frame_idx(f, frames)
        s_norm_img, s_raw_img, s_kp_2d = get_single_image_crop_demo(
                s_img,
                bboxes[idx],
                kp_2d=None,
                scale=1.1,
                crop_size=64)
        silhuette_imgs.append(s_raw_img[:,:,0])
        
        # 2. silhouette
        image_file_name = image_folder+'/'+str(frames[idx]).zfill(6)+'.png'
        img = cv2.cvtColor(cv2.imread(image_file_name), cv2.COLOR_BGR2RGB)
        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                img,
                bboxes[idx],
                kp_2d=None,
                scale=1.1,
                crop_size=224)
        norm_imgs.append(norm_img.detach().cpu().numpy())

        # 3. preprocessed bboxes & frames
        final_bboxes.append(bboxes[idx])
        final_frames.append(f)
    silhuette_imgs = np.array(silhuette_imgs)
    norm_imgs = np.array(norm_imgs)
    final_bboxes = np.array(final_bboxes)
    final_frames = np.array(final_frames)


    if len(final_frames) >= 15:
        break_id = final_frames[0:-1]+1 != final_frames[1:]
        if len(final_frames[np.where(break_id)[0]]) == 0:
            file_name = '/home/epinyoan/dataset/casia-b/dataset_b/all/crop/'+subject_id+'-'+condition+'-'+walk_id+'-'+angle+'.npz'
            np.savez(file_name, silhuette_imgs=silhuette_imgs, norm_imgs=norm_imgs, bboxes=final_bboxes, frames=final_frames)


if __name__ == '__main__':
    for folder_person in glob.glob('/home/epinyoan/dataset/casia-b/dataset_b/all/silhouettes/*'):
        subject_id = folder_person.split('/')[-1]
        # if int(subject_id) < 30:
        # if int(subject_id) >= 30 and int(subject_id) < 60:
        # if int(subject_id) >= 60 and int(subject_id) < 90:
        if int(subject_id) >= 90:
            for folder_condition in glob.glob(folder_person+'/*'):
                condition, walk_id = folder_condition.split('/')[-1].split('-')
                for folder_angle in glob.glob(folder_condition+'/*'):
                    angle = folder_angle.split('/')[-1]
                    load_video_crop(folder_angle, subject_id, condition, walk_id, angle)