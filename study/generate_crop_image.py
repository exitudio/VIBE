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

def load_video_crop(image_folder):
    subject_id, condition, walk_id, angle = image_folder.split('/')[-1].split('-')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
            
    person_id = list(tracking_results.keys())[0]
    bboxes = tracking_results[person_id]['bbox']
    frames = tracking_results[person_id]['frames']

    # sort file name
    image_file_names = [
        osp.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ]
    image_file_names = sorted(image_file_names)
    image_file_names = np.array(image_file_names)[frames]

    # load & crop
    silhuette_imgs = []
    norm_imgs = []
    pre_path = '/home/epinyoan/dataset/casia-b/dataset_b/all/silhouettes/'+subject_id+'/'+condition+'-'+walk_id+'/'+angle+'/'+subject_id+'-'+condition+'-'+walk_id+'-'+angle+'-'
    for idx, image_file_name in enumerate(image_file_names):
        silhouette_path = pre_path+str(frames[idx]).zfill(3)+'.png'
        s_img = cv2.imread(silhouette_path)
        if s_img is None:
            continue
        if idx != 0 and frames[idx-1] != frames[idx]-1:
            print(pre_path, frames[idx-1], frames[idx])
            return True

    #     s_norm_img, s_raw_img, s_kp_2d = get_single_image_crop_demo(
    #             s_img,
    #             bboxes[idx],
    #             kp_2d=None,
    #             scale=1.2,
    #             crop_size=64)
    #     silhuette_imgs.append(s_raw_img[:,:,0])
        
    #     img = cv2.cvtColor(cv2.imread(image_file_name), cv2.COLOR_BGR2RGB)
    #     norm_img, raw_img, kp_2d = get_single_image_crop_demo(
    #             img,
    #             bboxes[idx],
    #             kp_2d=None,
    #             scale=1.2,
    #             crop_size=224)
    #     norm_imgs.append(norm_img.detach().cpu().numpy())
    # silhuette_imgs = np.array(silhuette_imgs)
    # norm_imgs = np.array(norm_imgs)

    # # save
    # file_name = '/home/epinyoan/dataset/casia-b/dataset_b/all/crop/'+subject_id+'-'+condition+'-'+walk_id+'-'+angle+'.npz'
    # np.savez(file_name, silhuette_imgs=silhuette_imgs, norm_imgs=norm_imgs)

if __name__ == '__main__':
    for walk_folder in glob.glob('/home/epinyoan/dataset/casia-b/dataset_b/all/images/*'):
        out = load_video_crop(walk_folder)
        if out:
            print('--- broken ----')
            break