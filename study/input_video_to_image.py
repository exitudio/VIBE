from tqdm import tqdm
import glob
import os
import numpy as np

def save_images(file_name):
    crop_image_path = '/home/epinyoan/dataset/casia-b/dataset_b/all/crop_images/'
    folder_path = crop_image_path + file_name.split('/')[-1].split('.')[0]
    os.makedirs(folder_path, exist_ok=True)
    data = np.load(file_name, allow_pickle=True)
    for i, frame in enumerate(data['frames']):
        np.savez(folder_path+'/'+str(frame), silhuette_imgs=data['silhuette_imgs'][i], 
                        norm_imgs=data['norm_imgs'][i], 
                        bboxes=data['bboxes'][i])

if __name__ == '__main__':
    file_names = glob.glob('/home/epinyoan/dataset/casia-b/dataset_b/all/crop/*')
    all_file_names = []
    for file_name in file_names:
        subject_id = int(file_name.split('/')[-1].split('.')[0].split('-')[0])
        # if int(subject_id) < 30:
        # if int(subject_id) >= 30 and int(subject_id) < 60:
        # if int(subject_id) >= 60 and int(subject_id) < 90:
        if int(subject_id) >= 90:
            all_file_names.append(file_name)


    for file_name in tqdm(all_file_names):
        save_images(file_name)