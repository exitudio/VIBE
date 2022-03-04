#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://github.com/mkocabas/VIBE/releases/download/v0.2/vibe_data.zip"
unzip vibe_data.zip
rm vibe_data.zip
cd ..
mv data/vibe_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv data/vibe_data/yolov3.weights $HOME/.torch/models/
