# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
import torch.nn.functional as F
from mmengine.dist import get_dist_info

from mmengine import HOOKS
from mmengine.hooks import Hook
from mmdet.registry import HOOKS as MMHOOKS

@MODELS.register_module()
class ProjectionHead(nn.Module):
    def __init__(self, dim_in=[128,256,512], proj_dim=[128,256,512]):
        super(ProjectionHead, self).__init__()

        self.anchor0 = nn.Conv2d(proj_dim[0], 1, kernel_size=1, bias=False)
        self.anchor1 = nn.Conv2d(proj_dim[1], 1, kernel_size=1, bias=False)
        self.anchor2 = nn.Conv2d(proj_dim[2], 1, kernel_size=1, bias=False)


        self.proj0 = nn.Sequential(
            nn.Conv2d(dim_in[0], dim_in[0], kernel_size=1),
            nn.SyncBatchNorm(dim_in[0]),
            nn.ReLU(),
            nn.Conv2d(dim_in[0], proj_dim[0], kernel_size=1)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim_in[1], dim_in[1], kernel_size=1),
            nn.SyncBatchNorm(dim_in[1]),
            nn.ReLU(),
            nn.Conv2d(dim_in[1], proj_dim[1], kernel_size=1)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim_in[2], dim_in[2], kernel_size=1),
            nn.SyncBatchNorm(dim_in[2]),
            nn.ReLU(),
            nn.Conv2d(dim_in[2], proj_dim[2], kernel_size=1)
        )

    def forward(self, x):
        x0 = self.proj0(x[0])
        x0 = self.anchor0(x0)
        x1 = self.proj1(x[1])
        x1 = self.anchor1(x1)
        x2 = self.proj2(x[2])
        x2 = self.anchor2(x2)
        return x0, x1, x2
class SharedProjectionHead(nn.Module):
    def __init__(self, dim_in=[128,256,512], proj_dim=256):
        super().__init__()
        self.proj0 = nn.Sequential(
            nn.Conv2d(dim_in[0], dim_in[0], kernel_size=1),
            nn.SyncBatchNorm(dim_in[0]),
            nn.ReLU(),
            nn.Conv2d(dim_in[0], proj_dim, kernel_size=1)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim_in[1], dim_in[1], kernel_size=1),
            nn.SyncBatchNorm(dim_in[1]),
            nn.ReLU(),
            nn.Conv2d(dim_in[1], proj_dim, kernel_size=1)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim_in[2], dim_in[2], kernel_size=1),
            nn.SyncBatchNorm(dim_in[2]),
            nn.ReLU(),
            nn.Conv2d(dim_in[2], proj_dim, kernel_size=1)
        )

    def forward(self, img_feats):  # img_feats: (P3, P4, P5)
        z0 = self.proj0(img_feats[0])  # [B, C0, H0, W0]
        z1 = self.proj1(img_feats[1])  # [B, C1, H1, W1]
        z2 = self.proj2(img_feats[2])  # [B, C2, H2, W2]
        return [z0, z1, z2]

class MoESharedProjectionHead(nn.Module):
    """
    多尺度 MoE 投影器：
      - 每个尺度先做同构的预投影(Conv-BN-ReLU)
      - 进入 E=k×N 个专家，每个专家输出 proj_dim//k 通道
      - 路由器：Conv1x1 -> softmax，按位置 Top-k 选择专家并拼接
      - 返回: [z0, z1, z2], 以及每尺度的 {Fi, Pi} 统计 (用于负载均衡)
    """
    def __init__(self, dim_in=[128, 256, 512], proj_dim=256, k=2, N=8, lb_coef=1e-3):
        super().__init__()
        assert proj_dim % k == 0, "proj_dim 必须能被 k 整除"
        self.k, self.N = k, N
        self.E = k * N
        self.chunk = proj_dim // k
        self.lb_coef = lb_coef

        # 预投影 (与原 SharedProjectionHead 一致)
        self.pre0 = nn.Sequential(
            nn.Conv2d(dim_in[0], dim_in[0], kernel_size=1),
            nn.SyncBatchNorm(dim_in[0]),
            nn.ReLU(),
        )
        self.pre1 = nn.Sequential(
            nn.Conv2d(dim_in[1], dim_in[1], kernel_size=1),
            nn.SyncBatchNorm(dim_in[1]),
            nn.ReLU(),
        )
        self.pre2 = nn.Sequential(
            nn.Conv2d(dim_in[2], dim_in[2], kernel_size=1),
            nn.SyncBatchNorm(dim_in[2]),
            nn.ReLU(),
        )

        # 专家卷积: 每个尺度一组专家，每个专家输出 proj_dim//k 通道
        self.experts0 = nn.ModuleList([nn.Conv2d(dim_in[0], self.chunk, 1) for _ in range(self.E)])
        self.experts1 = nn.ModuleList([nn.Conv2d(dim_in[1], self.chunk, 1) for _ in range(self.E)])
        self.experts2 = nn.ModuleList([nn.Conv2d(dim_in[2], self.chunk, 1) for _ in range(self.E)])

        # 路由器：轻量 1×1 conv 产生 E 个 logits（逐位置路由）
        self.router0 = nn.Conv2d(dim_in[0], self.E, 1, bias=True)
        self.router1 = nn.Conv2d(dim_in[1], self.E, 1, bias=True)
        self.router2 = nn.Conv2d(dim_in[2], self.E, 1, bias=True)

        # 简单稳定初始化：让初期偏向选择固定的 k 个专家，避免训练初期震荡
        with torch.no_grad():
            for r in [self.router0, self.router1, self.router2]:
                nn.init.zeros_(r.weight)
                nn.init.constant_(r.bias, 0.0)

    @torch.no_grad()
    def _topk_mask(self, scores: torch.Tensor, k: int):
        """scores: [B, E, H, W] -> 返回 one-hot mask: [B, E, H, W, k] (每个位置 top-k 的 one-hot)"""
        B, E, H, W = scores.shape
        probs = torch.softmax(scores, dim=1)  # [B,E,H,W]
        topk = torch.topk(probs, k, dim=1, largest=True, sorted=False)  # values, indices: [B,k,H,W]
        mask = torch.zeros(B, E, H, W, k, device=scores.device, dtype=scores.dtype)
        mask.scatter_(1, topk.indices.unsqueeze(-1), 1.0)  # 在 expert 维度 one-hot
        return probs, mask  # probs 用于统计，mask 用于选择

    def _route_and_fuse(self, x, pre, experts, router):
        """
        x: [B,C,H,W] 输入特征
        pre: 预投影 block
        experts: ModuleList[Conv2d], 长度 E
        router: Conv2d -> [B,E,H,W]
        返回:
            z: [B, proj_dim, H, W]
            stats: dict(F=[E], P=[E]) 该尺度的负载统计
        """
        B, C, H, W = x.shape
        x = pre(x)  # 预投影
        logits = router(x)                     # [B,E,H,W]
        probs, mask = self._topk_mask(logits, self.k)  # probs: [B,E,H,W], mask: [B,E,H,W,k]

        # 逐专家前向（向量化融合，避免 Python for 循环瓶颈）
        # 把所有专家参数堆叠，做 grouped 卷积
        # 这里为了简洁，采用小循环；如果追求极致速度，可用自定义 grouped GEMM 实现
        z_chunks = []
        active_counts = torch.clamp(mask.sum(dim=(-1)), min=1e-6)  # [B,E,H,W] 每位置被选择的专家总数(=k)
        for e, conv in enumerate(experts):
            ze = conv(x)  # [B, chunk, H, W]
            # 该专家在 top-k 被选中的 one-hot（把 k 维相加）
            me = mask[:, e:e + 1].sum(dim=-1)  # 形状 [B, 1, H, W]
            z_chunks.append(ze * me)  # ze: [B, chunk, H, W]

        # 拼接 k 个被选专家 → proj_dim 通道
        z = torch.cat(z_chunks, dim=1)  # [B, E*chunk, H, W]
        # 仅保留被选中的 k 个 chunk：用一个简单的通道重排近似（更高效可写成索引收集）
        # E*chunk = (kN)*(proj_dim//k) = N*proj_dim
        # 为确保输出为 proj_dim，这里做一个按块的分组求和（每 N 个专家映射到同一 chunk 组）
        z = z.view(B, self.N, self.k * self.chunk, H, W).sum(dim=1)  # [B, k*chunk, H, W] = [B, proj_dim, H, W]

        # 负载统计：Fi=每个专家处理的 token 占比；Pi=平均路由概率
        # token 数量 = B*H*W
        tokens = float(B * H * W)
        Fi = probs.mean(dim=(0, 2, 3))  # [E] 平均“软”分配比例
        # 严格按选择统计（one-hot）：被选中的比例
        hard_sel = mask.sum(dim=-1).sum(dim=(0, 2, 3)) / (tokens)  # [E]
        Pi = Fi  # 论文中常用 Fi, Pi 的乘积；这里 Pi 复用 softmax 概率平均
        stats = dict(F=hard_sel.detach(), P=Pi.detach())
        return z, stats

    def forward(self, img_feats, return_aux=False):
        """
        img_feats: (P3, P4, P5) = [B,Ci,Hi,Wi]
        return: [z0,z1,z2], aux(optional)
        """
        z0, s0 = self._route_and_fuse(img_feats[0], self.pre0, self.experts0, self.router0)
        z1, s1 = self._route_and_fuse(img_feats[1], self.pre1, self.experts1, self.router1)
        z2, s2 = self._route_and_fuse(img_feats[2], self.pre2, self.experts2, self.router2)
        if not return_aux:
            return [z0, z1, z2]
        # 负载均衡损失（分尺度求和）
        aux = self.compute_load_balancing_loss([s0, s1, s2])
        return [z0, z1, z2], aux

    def compute_load_balancing_loss(self, stats_list):
        # L_aux = (kN) * Σ_i Fi * Pi
        losses = []
        for stats in stats_list:
            F, P = stats['F'], stats['P']  # [E], [E]
            losses.append(self.E * (F * P).sum())
        return sum(losses) / len(losses) * self.lb_coef


@MODELS.register_module()
class OWODDetector(YOLODetector):
    """Implementation of Open-World YOLO"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 num_prev_classes: int = 0,
                 num_prompts: int = 80,
                 prompt_dim: int = 512,
                 embedding_path: str = '',
                 unknown_embedding_path: str = '',
                 anchor_embedding_path: str = '',
                 embedding_mask: Union[List, int] = None,
                 freeze_prompt: bool = False,
                 use_mlp_adapter: bool = False,
                 use_MSCAL: bool = False,
                 use_SPGA: bool = False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.num_prev_classes = num_prev_classes
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.use_MSCAL = use_MSCAL
        self.use_SPGA = use_SPGA
        self.unknown_index = self.num_training_classes - 2
        self.temperature = 0.5
        self.tmp_labels = None
        self.alpha = 0
        self.beita = -1
        super().__init__(*args, **kwargs)
        if self.use_MSCAL:
            self.projectors = nn.ModuleList([ProjectionHead() for i in range(self.unknown_index)])
        if self.use_SPGA:
            self.projection = SharedProjectionHead()
            # self.projection = MoESharedProjectionHead(dim_in=[128, 256, 512], proj_dim=256, k=2, N=4, lb_coef=1e-2)
            self.prototypes = nn.Parameter(torch.randn(self.unknown_index, 256))

        if self.training:
            self._initialize_embeddings(embedding_path, unknown_embedding_path, anchor_embedding_path)
        if self.freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if embedding_mask:
            if isinstance(embedding_mask, int):
                self._grad_mask = torch.ones(num_train_classes, dtype=torch.bool)[:, None]
                self._grad_mask[:embedding_mask] = False
            else:
                self._grad_mask = torch.Tensor(embedding_mask).bool()[:, None]
            assert len(self._grad_mask) == num_train_classes
            self.embeddings.register_hook(lambda grad: grad * self._grad_mask.to(grad.device))#embeddings在backward()时，只对True位置更新梯度

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim))
        else:
            self.adapter = None
        if self.use_MSCAL:
            self.enable_projector_grad(self.num_prev_classes)
        # for name, param in self.named_parameters():
        #     print(f"参数: {name} | 可训练: {param.requires_grad}")

    def _initialize_embeddings(self,embedding_path, unknown_embedding_path, anchor_embedding_path):
        if len(embedding_path) > 0:
            if 'owodb' in embedding_path:
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float()).unsqueeze(0)
            else:
                self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
        else:
            # random init
            embeddings = nn.functional.normalize(torch.randn(
                (self.num_training_classes, self.prompt_dim)),
                                                    dim=-1)
            self.embeddings = nn.Parameter(embeddings)
        known_embeddings = self.embeddings
        if len(unknown_embedding_path) > 0:
            unknown_embeddings = nn.Parameter(torch.from_numpy(
                np.load(unknown_embedding_path)).float()).unsqueeze(0)
            # self.embeddings = nn.Parameter(torch.cat([known_embeddings, unknown_embeddings], dim=1))
            # 将微调后的object文本嵌入减掉已知类文本嵌入的均值，可以认为是未知类的嵌入
            # if self.alpha != 0:
            #     normalized_embedding = F.normalize(known_embeddings, p=2, dim=2)
            #     normalized_embedding = normalized_embedding.mean(dim=1)
            #     unknown_embeddings = unknown_embeddings - self.alpha * normalized_embedding
        if len(anchor_embedding_path) > 0:
            if isinstance(anchor_embedding_path, dict):#融合微调后的object thing entity的embedding
                object_embeddings = nn.Parameter(torch.from_numpy(
                    np.load(anchor_embedding_path['object'])).float()).unsqueeze(0)
                thing_embeddings = nn.Parameter(torch.from_numpy(
                    np.load(anchor_embedding_path['thing'])).float()).unsqueeze(0)
                entity_embeddings = nn.Parameter(torch.from_numpy(
                    np.load(anchor_embedding_path['entity'])).float()).unsqueeze(0)
                stack_embeddings = torch.stack([object_embeddings,
                                                thing_embeddings,
                                                entity_embeddings], dim=0)
                anchor_embeddings = stack_embeddings.mean(dim=0)
                # self.embeddings = nn.Parameter(torch.cat([self.embeddings, anchor_embeddings], dim=1))

            else:#仅有微调后的object的embedding
                anchor_embeddings = nn.Parameter(torch.from_numpy(
                    np.load(anchor_embedding_path)).float()).unsqueeze(0)
            self.embeddings = nn.Parameter(torch.cat([self.embeddings, anchor_embeddings], dim=1))

            # 将微调后的object文本嵌入减掉已知类文本嵌入的均值，可以认为是未知类的嵌入
            generic_embedding =self.embeddings[:,self.unknown_index, :]
            if self.alpha != 0:
                normalized_embedding = F.normalize(known_embeddings,p=2,dim=2)
                normalized_embedding = normalized_embedding.mean(dim=1)
                generic_embedding = generic_embedding - self.alpha * normalized_embedding
            if self.beita != 0:
                # 2. 动态正交化 (使用稳定的SVD)
                U, S, Vh = torch.linalg.svd(known_embeddings, full_matrices=False)
                k = min(20, Vh.size(0))  # 取前20个主成分
                orthogonal_component = Vh[:k]
                # 计算投影并减去
                projection = torch.sum(orthogonal_component * generic_embedding.unsqueeze(0),
                                       dim=1, keepdim=True).sum(dim=0)
                generic_embedding = generic_embedding - projection * self.beita

            embeddings = torch.cat([known_embeddings, unknown_embeddings],dim=1)
            embeddings = torch.cat([embeddings, generic_embedding.unsqueeze(0)],dim=1).squeeze(0)
            self.embeddings = nn.Parameter(embeddings)
        elif len(unknown_embedding_path) > 0:
            embeddings = torch.cat([known_embeddings, unknown_embeddings],dim=1).squeeze(0)
            self.embeddings = nn.Parameter(embeddings)

    def update_embeddings(self, embeddings):
        # update embeddings when loading from checkpoint
        prev_embeddings = embeddings[:self.num_prev_classes]
        cur_embeddings = self.embeddings[self.num_prev_classes:].detach().cpu()
        embeddings = torch.cat([prev_embeddings, cur_embeddings], dim=0)
        return embeddings

    def enable_projector_grad(self,index):
        for i in range(index, self.unknown_index):
            for param in self.projectors[i].parameters():
                param.requires_grad = True
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        # inputs = batch_inputs[0]
        # image = inputs.permute(1, 2, 0).cpu().numpy()
        # image = (image * 255).astype(np.uint8)
        # import matplotlib.pyplot as plt
        # image = image.astype(np.uint8)
        # plt.imshow(image)
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        # self.visualize(batch_inputs,batch_data_samples)
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats, flatten_scores_list = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses,tmp_labels = self.bbox_head.loss(img_feats, txt_feats,
                                        batch_data_samples)
        self.tmp_labels =  tmp_labels

        # losses['loss_wildcard_align'] = self.wildcard_alignment_loss(img_feats, self.tmp_labels)
        if self.use_MSCAL or self.use_SPGA:
            contrastive_loss = self.contrastive_loss(flatten_scores_list)
            losses.update({'contrastive_loss': contrastive_loss})
        # 合并 MoE 负载均衡正则（仅在 use_SPGA & MoE 投影器时存在）
        # if hasattr(self, "_moe_aux") and (self._moe_aux is not None):
        #     losses['loss_moe_aux'] = self._moe_aux

        return losses
    def contrastive_loss(self,
            flatten_scores_list: Tensor) -> Tensor:
        """Calculate the contrastive loss."""
        #mask = torch.zeros_like(flatten_scores_list)
        #mask = mask.to(torch.bool)
        #mask.scatter_(2, self.tmp_labels.unsqueeze(2), 1)
        b, n, c = flatten_scores_list.shape
        flatten_scores_list = torch.div(flatten_scores_list,
                                        self.temperature)
        contrastive_losses = []
        for i in range(c):
            mask = (self.tmp_labels == i)
            positive = flatten_scores_list[:,:,i][mask]
            if not positive.numel() > 0:
                continue
            negative = flatten_scores_list[:,:,i][~mask]
            positive_exp = torch.exp(positive)
            negative = torch.exp(negative)
            log_prob = positive - torch.log(positive_exp.sum() + negative.sum())
            contrastive_losses.append(-log_prob.sum()/mask.sum())

        return sum(contrastive_losses) / len(contrastive_losses)

    def wildcard_alignment_loss(self, img_feats, tmp_labels):
        """
        img_feats: (B, C, H, W) or List/Tuple[(B, C, H, W)]
        tmp_labels: (B, H, W)
        """
        if isinstance(img_feats, (tuple, list)):
            img_feats = img_feats[-1]  # 取最后一层特征

        # Normalize features
        img_feats = F.normalize(img_feats, dim=1)  # (B, C, H, W)

        # wildcard embedding
        wildcard_embed = self.embeddings[-2]  # shape: (C,)
        wildcard_embed = F.normalize(wildcard_embed, dim=0).view(1, -1, 1, 1)  # (1, C, 1, 1)

        # Cosine similarity
        cosine_sim = F.cosine_similarity(img_feats, wildcard_embed, dim=1)  # (B, H, W)

        with torch.no_grad():
            unknown_mask = (tmp_labels == self.num_training_classes - 2).float()  # (B, H, W)

        loss = - (cosine_sim * unknown_mask).sum() / (unknown_mask.sum() + 1e-6)
        return loss
    def visualize(self, batch_inputs, batch_data_samples):
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        # 将数据转到CPU并转换为numpy格式
        image_tensor = batch_inputs[0].detach().cpu()  # 取batch中的第一个样本
        bboxes_np = batch_data_samples['bboxes_labels'].detach().cpu().numpy()

        # 转换图像张量格式 [C, H, W] -> [H, W, C]
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # 反归一化（如果输入做了归一化需要加上这个）
        # image_np = (image_np * std + mean) * 255  # 替换为你的归一化参数
        # 如果没有做归一化，直接转换到0-255范围
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        # 创建画布
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image_np)
        # 遍历所有边界框
        for bbox in bboxes_np:
            class_label = int(bbox[1])  # 第二个元素是类别标签
            x1, y1, x2, y2 = bbox[2:]  # 后四个元素是坐标
            if bbox[0] == 0:
                # 绘制矩形框
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                # 添加类别标签文本
                plt.text(
                    x1, y1 - 5, str(class_label),
                    color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.7, pad=0, edgecolor='none')
                )

        plt.axis('off')
        plt.show()
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, flatten_scores_list = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes

        results_list = self.bbox_head.predict(img_feats,
                                                txt_feats,
                                                batch_data_samples,
                                                rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)
        # # 2. 应用未知性注意力
        # attn_feats = []
        # for i, feat in enumerate(img_feats):
        #     attn_feats.append(self.attentions[i](feat))
        # img_feats = attn_feats

        # use embeddings
        txt_feats = self.embeddings[None]
        if isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']

        if self.adapter is not None:
            # txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats_first21 = txt_feats[:, :21, :]  # (1, 21, 512)
            adapted_first21 = self.adapter(txt_feats_first21) + txt_feats_first21 # (1, 21, 512)
            # adapted_first21 = nn.functional.normalize(adapted_first21, dim=-1, p=2)
            txt_feats_last = txt_feats[:, 21:, :]  # (1, 1, 512)
            txt_feats = torch.cat([adapted_first21, txt_feats_last], dim=1)
            # txt_feats = self.adapter(txt_feats)
            # txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)#融合图像和文本特征
            else:
                img_feats = self.neck(img_feats)
                # img_feats = self.img_adapter(img_feats)

        if self.use_MSCAL:
            flatten_scores_list = []
            for projector in self.projectors:  # 遍历每个投影器
                scores = projector(img_feats)
                flatten_scores = [
                    score.permute(0, 2, 3, 1).reshape(img_feats[0].shape[0], -1)
                    for score in scores
                ]
                flatten_scores = torch.cat(flatten_scores, dim=1)
                flatten_scores_list.append(flatten_scores)

            flatten_scores_list = torch.stack(flatten_scores_list, dim=2)
            return img_feats, txt_feats, flatten_scores_list
        # if self.use_SPGA:
        #     projected_feats, moe_aux = self.projection(img_feats, return_aux=True)  # list of [B,C,H,W], scalar
        #     self._moe_aux = moe_aux  # 暂存，loss() 中读取
        #     flatten_feats = []
        #     for feats in projected_feats:
        #         b, c, h, w = feats.shape
        #         feat_flat = feats.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, N, C]
        #         flatten_feats.append(feat_flat)
        #     z = torch.cat(flatten_feats, dim=1)  # [B, N, D]
        #     z = F.normalize(z, dim=-1)
        #     prototypes = F.normalize(self.prototypes, dim=-1)  # [C, D]
        #     scores = torch.einsum("bnd,cd->bnc", z, prototypes)  # [B, N, C]
        #     return img_feats, txt_feats, scores

        if self.use_SPGA:
            flatten_scores_list = []
            projected_feats = self.projection(img_feats)  # list of [B, C, H, W]
            flatten_feats = []
            for feats in projected_feats:
                b, c, h, w = feats.shape
                feat_flat = feats.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, N, C]
                flatten_feats.append(feat_flat)
            z = torch.cat(flatten_feats, dim=1)  # [B, N, D]
            z = F.normalize(z, dim=-1)  # optional: 余弦相似度标准化
            prototypes = F.normalize(self.prototypes, dim=-1)  # [C, D]

            # 相似度计算（点积或余弦）
            scores = torch.einsum("bnd,cd->bnc", z, prototypes)  # [B, N, C]
            return img_feats, txt_feats, scores
        return img_feats, txt_feats, None
