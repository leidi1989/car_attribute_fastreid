'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-30 11:03:41
LastEditors: Leidi
LastEditTime: 2021-04-30 14:07:53
'''
import logging
import re
import sys
# print('\n', sys.path, '\n')

sys.path.append('.')

import torch
from torchvision import transforms
import cv2

from projects.FastAttr.fastattr import *
from fastreid.utils import comm
from fastreid.data.transforms import build_transforms
from fastreid.data.build import _root, build_reid_train_loader, build_reid_test_loader
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.engine import DefaultTrainer
from fastreid.config import get_cfg


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
    
    image_path = r'/home/leidi/Dataset/CompCars_analyze/data/changan_1cb8c360d2adec.jpg'
    pretrain_path = r'/home/leidi/Workspace/car_attribute_fastreid/projects/FastAttr/logs/compcars/strong_baseline/model_best.pth'
    cfg = setup(args)
    
    # 读取模型及对应与训练权重
    model = AttrTrainer.build_model(cfg)    # 按照配置文件构建模型
    Checkpointer(model).load(pretrain_path)
    model.eval()
    
    # 读取图片
    transform = transforms.ToTensor()
    image = cv2.imread(image_path)
    image = transform(image).unsqueeze(0).cuda()
    # inference
    result = model(image)
    print(result)
    print(result.shape)
    # if args.eval_only:
    #     cfg.defrost()
    #     cfg.MODEL.BACKBONE.PRETRAIN = True
    #     model = AttrTrainer.build_model(cfg)

    #     pretrain_path = r'/home/leidi/Workspace/car_attribute_fastreid/projects/FastAttr/logs/compcars/strong_baseline/model_best.pth'

    #     Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    #     res = AttrTrainer.test(cfg, model)
    #     return res

    # trainer = AttrTrainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()


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
