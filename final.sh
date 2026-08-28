#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate Halo

python inference_ourmethod_final.py --video_path ./assets/bmx-trees.mp4 --prompt "Leopard running up a snowy hill in a forest"  \
        --video_length 24 \
        --output_path tmp_result --config_path ./configs/guidance_config_inject12.yaml