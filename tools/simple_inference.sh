#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

python3 submit_dance.py ${args} --exp_name tracker --resume $1

# Read args from beach_volleyball.args
args=$(cat configs/beach_volleyball.args)

# Run inference
python3 submit_dance.py ${args} \
    --resume ./weights/motrv2_dancetrack.pth \
    --mot_path /data/Dataset/mot \
    --output_dir outputs \
    --exp_name beach_volleyball_tracker