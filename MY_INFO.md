1. source
```conda activate vibe-env```


```python3 demo.py --vid_file /home/epinyoan/dataset/casia-b/dataset_b/Datset-B-2/video/124-nm-06-090.avi --output_folder output/ --tracker_batch_size 2 --vibe_batch_size 64```





```python3 train.py --cfg configs/config.yaml```

To use pose [args.tracking_method = 'pose'] need to install STAFF
https://github.com/mkocabas/VIBE/issues/33

1. AMASS 
Tar ```tar -xf archive.tar.bz2```
Tried [Body] of SMPL-X Gender Neutral & SMPL-X Gender Specific - don't work



