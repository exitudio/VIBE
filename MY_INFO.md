folder
1. crop images based on the intersection of silhouette & object detection
    Use silhouette frames as the anchor, if no object detected use the bbox of the adjacent frames.
```~/dataset/casia-b/dataset_b/all/crop/```
```python3 generate_crop_image.py```

2. break apart .npz for the whole video to single .npz (image) for optimizing dataloader (Actually, this file can be combined into the first file)
```~/dataset/casia-b/dataset_b/all/crop_images/```
```python3 input_video_to_image.py```

3. Train silhouette
```python3 train_silhouette.py```


Some of video cannot detect object [a.k.a human] (less than 25 frames) ls in here
```/home/epinyoan/git/VIBE/study/not_detect.txt```

########################################
########################################
RUN DEMO

1. source
```conda activate vibe-env```
2. run demo
```python3 demo.py --vid_file /home/epinyoan/dataset/casia-b/dataset_b/Datset-B-2/video/124-nm-06-090.avi --output_folder output/ --tracker_batch_size 2 --vibe_batch_size 64```





```python3 train.py --cfg configs/config.yaml```

To use pose [args.tracking_method = 'pose'] need to install STAFF
https://github.com/mkocabas/VIBE/issues/33

1. AMASS 
Tar ```tar -xf archive.tar.bz2```
Tried [Body] of SMPL-X Gender Neutral & SMPL-X Gender Specific - don't work



