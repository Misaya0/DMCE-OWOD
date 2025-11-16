# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Sequence
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmyolo.registry import HOOKS

@HOOKS.register_module()
class visualize_attention_Hook(Hook):
    def __init__(self,):
        pass
    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        if epoch not in self.update_epochs or epoch <= self.last_updated:
            return

        model = runner.model
        # 获取嵌入矩阵
        embeddings = model.embeddings.data.clone()
        num_known = self.num_known_classes

        # 应用动态正交化
        known_emb = embeddings[:num_known]
        object_finetune = embeddings[-1].clone()  # 最后一个位置是object(微调)

        # 2. 动态正交化 (使用稳定的SVD)
        U, S, Vh = torch.linalg.svd(known_emb, full_matrices=False)
        k = min(20, Vh.size(0))  # 取前20个主成分
        orthogonal_component = Vh[:k]

        # 计算投影并减去
        projection = torch.sum(orthogonal_component * object_finetune.unsqueeze(0),
                              dim=1, keepdim=True).sum(dim=0)
        object_enhanced = object_finetune - projection

        # 创建更新后的嵌入矩阵
        updated_embeddings = embeddings.clone()
        updated_embeddings[-1] = object_enhanced

        # 应用更新
        model.embeddings.data = updated_embeddings
        self.last_updated = epoch

        # 日志记录
        runner.logger.info(f"正交化更新完成 (epoch {epoch})")
    def after_train_epoch(self, runner: Runner):

        model = runner.model

