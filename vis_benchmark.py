# -*- coding: utf-8 -*-
# @Description :
# @Author      : Guocheng Qian
# @Email       : guocheng.qian@kaust.edu.sa

import os
import sys
import os.path as osp
from tf_lib.pc_viz import visualize_pointcloud
import argparse

parser = argparse.ArgumentParser(description='Qualitative comparision of ResGCN '
                                             'against PlainGCN on PartNet segmentation')

parser.add_argument('--pointsize', default=3, type=int, help='point size')
parser.add_argument('--input_path',
                    default=None,
                    type=str,
                    help='path to the input')
parser.add_argument('--gt_path',
                    default=None,
                    type=str,
                    help='path to the gt')
parser.add_argument('--pugcn_path',
                    default=None,
                    type=str, help='path to the result')
parser.add_argument('--png_path',
                    default=None, required=True,
                    type=str, help='name of file to show')
parser.add_argument('--filename',
                    default=None, required=True,
                    type=str, help='name of file to show')

args = parser.parse_args()

comparison_folder_list = [args.input_path]
text = ['Input']

if args.pugcn_path is not None:
    comparison_folder_list.append(args.pugcn_path)
    text.append('pugcn_path')
if args.gt_path is not None:
    comparison_folder_list.append(args.gt_path)
    text.append('Ground Truth')

filename = args.filename
png_path = osp.join(args.png_path, filename.replace(filename.split('.')[-1], 'png'))
# color = [22, 139, 119]
color = [0, 0, 0]

visualize_pointcloud(comparison_folder_list,
                     filename,
                     color,
                     text=text,
                     png_path=png_path,
                     interactive=True,
                     orientation='horizontal',
                     pointsize=args.pointsize
                     )
