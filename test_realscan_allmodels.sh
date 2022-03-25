#!/usr/bin/env bash
conda activate pcu

# kitti
bash test_realscan.sh pretrain/pugan-pugan/ ./data/real_scan_kitti_pugcn 0 --model pugan --more_up 2
bash test_realscan.sh pretrain/pugan-pugc/  ./data/real_scan_kitti_pugcn 0 --model pugcn --k 20
bash test_realscan.sh pretrain/pugan-pugan/ ./data/real_scan_kitti_pugcn 0 --model enhanced_PCU
\