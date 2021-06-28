'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-30 11:03:41
LastEditors: Leidi
LastEditTime: 2021-05-11 10:10:23
'''
import sys
sys.path.append('.')

from projects.FastAttr.fastattr import *
from fastreid.utils import comm
from fastreid.data.transforms import build_transforms
from fastreid.data.build import _root, build_reid_train_loader, build_reid_test_loader
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.engine import DefaultTrainer
from fastreid.config import get_cfg
import numpy as np
import cv2
from torchvision import transforms
import torch
import logging
import re

import os
from numpy.core.fromnumeric import argmax
import copy

# print('\n', sys.path, '\n')




class AttrTrainer(DefaultTrainer):
    sample_weights = None

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = DefaultTrainer.build_model(cfg)
        if cfg.MODEL.LOSSES.BCE.WEIGHT_ENABLED and \
                AttrTrainer.sample_weights is not None:
            setattr(model, "sample_weights",
                    AttrTrainer.sample_weights.to(model.device))
        else:
            setattr(model, "sample_weights", None)
        return model

    @classmethod
    def build_train_loader(cls, cfg):

        logger = logging.getLogger("fastreid.attr_dataset")
        train_items = list()
        attr_dict = None
        for d in cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(
                root=_root, combineall=cfg.DATASETS.COMBINEALL)
            if comm.is_main_process():
                dataset.show_train()
            if attr_dict is not None:
                assert attr_dict == dataset.attr_dict, f"attr_dict in {d} does not match with previous ones"
            else:
                attr_dict = dataset.attr_dict
            train_items.extend(dataset.train)

        train_transforms = build_transforms(cfg, is_train=True)
        train_set = AttrDataset(train_items, train_transforms, attr_dict)

        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        AttrTrainer.sample_weights = data_loader.dataset.sample_weights
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        attr_dict = dataset.attr_dict
        if comm.is_main_process():
            dataset.show_test()
        test_items = dataset.test

        test_transforms = build_transforms(cfg, is_train=False)
        test_set = AttrDataset(test_items, test_transforms, attr_dict)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, AttrEvaluator(cfg, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    # 所需文件路径
    attribute_list_path = r'/home/leidi/Dataset/CompCars_6_classes_analyze/ImageSets/encode_list.txt'
    detect_source_path = r'/home/leidi/Dataset/CompCars_6_classes_analyze/data'
    pretrain_path = r'/home/leidi/Workspace/car_attribute_fastreid/projects/FastAttr/logs/compcars_20210511/strong_baseline/model_best.pth'
    thres = 0.7
    tpye_split_key = 'mpv'
    
    # 获取属性列表
    attribute_list = []
    with open(attribute_list_path, 'r') as f:
        for n in f.readlines():
            attribute_list.append(n.strip('\n'))
    tpye_split = attribute_list.index(tpye_split_key)
    attribute_make_id_list = attribute_list[0: tpye_split]
    attribute_car_type_id_list = attribute_list[tpye_split:]

    # 读取模型及对应与训练权重
    cfg = setup(args)
    model = AttrTrainer.build_model(cfg)    # 按照配置文件构建模型
    Checkpointer(model).load(pretrain_path)
    model.eval()

    # 读取图片
    for one_picture in os.listdir(detect_source_path):
        image_path = os.path.join(detect_source_path, one_picture)
        transform = transforms.ToTensor()
        image = cv2.imread(image_path)
        image = transform(image).unsqueeze(0).cuda()

        # inference
        result = model(image).cpu().detach().numpy()
        pred_labels = copy.copy(result)
        pred_labels[result < thres] = 0
        pred_labels[result >= thres] = 1

        # decode
        result_list = pred_labels[0]
        result_make_id_list = result_list[0:tpye_split]
        result_car_type_list = result_list[tpye_split:]
        make_id = np.where(result_make_id_list == 1)
        car_type = np.where(result_car_type_list == 1)

        make_id_list = []
        car_list = []
        for n in range(0, len(make_id[0])):
            make_id_list.append(attribute_make_id_list[make_id[0][n]])

        for n in range(0, len(car_type[0])):
            car_list.append(attribute_car_type_id_list[car_type[0][n]])

        print('\n', one_picture, ':')
        print(make_id_list, car_list)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
