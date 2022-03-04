#!/usr/bin/env bash

mkdir -p ./data/vibe_db
export PYTHONPATH="./:$PYTHONPATH"

# AMASS
python3 lib/data_utils/amass_utils.py --dir ./data/amass

# InstaVariety
# Comment this if you already downloaded the preprocessed file
python3 lib/data_utils/insta_utils.py --dir ./data/vibe_db

# 3DPW
python3 lib/data_utils/threedpw_utils.py --dir ./data/3dpw

# MPI-INF-3D-HP
python3 lib/data_utils/mpii3d_utils.py --dir ./data/mpi_inf_3dhp

# PoseTrack
python3 lib/data_utils/posetrack_utils.py --dir ./data/posetrack

# PennAction
python3 lib/data_utils/penn_action_utils.py --dir ./data/penn_action
