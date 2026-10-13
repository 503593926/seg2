import torch
from torch import nn
from torch.nn import functional as F

from segmodel.modules.backbone import ConvNeXtUnet, Stem, init_convnext_backbone
from segmodel.modules.conditioning import PFMDecomposer
from segmodel.modules.tissue_states import TissueStateDecoder

def _mask_pfm_features(mask, *feature_groups):
    """把被屏蔽 PFM 的 f1/f2 清零，返回与输入同结构的元组。

    必须清零而不是只压低路由 logits：Joint Router 的输入包含每个 PFM 的 f1，若这一路
    仍带有效数值，联合 MLP 可以让它影响其他 expert 的 logits——等于模型仍间接用到了被
    声明为不可用的信息。清零后它的加权贡献恒为 0，投影层在这一步也拿不到梯度。
    """
    groups = []
    for features in feature_groups:
        zeroed = []
        for index, feature in enumerate(features):
            keep = mask[:, index].reshape(-1, *([1] * (feature.dim() - 1)))
            zeroed.append(feature * keep.to(feature.dtype))
        groups.append(tuple(zeroed))
    return groups


class SegmentationModel(nn.Module):
    gene_names: list[str]  # 由 pipeline 按当前训练 fold 的基因顺序写入。

    def __init__(
        self,
        pfm_patch_dims,
        pfm_cls_dims,
        propagation_channels=4,
        spatial_pool=4,
        top_k=2,
        encoder_chans=(96, 128, 160),
        decoder_chans=(128, 96),
        global_dim=256,
        spatial_dim=32,
        skip_drop_path=0.1,
        drop_path=(0.0, 0.0, 0.0),
        backbone_init="pytorch",
        gene_count=0,
        joint_prediction=False,
        phenotype_dim=16,
        molecular_dim=64,
        detach_phenotype=False,
        use_seg_condition=True,
        use_gene_feedback=True,
        tissue_state_count=0,
        tissue_state_dim=96,
    ):
        super().__init__()
        if backbone_init not in ("pytorch", "convnext"):
            raise ValueError(f"Unknown backbone_init: {backbone_init!r}; use 'pytorch' or 'convnext'")
        self.pfm_patch_dims = tuple(pfm_patch_dims)
        self.pfm_cls_dims = tuple(pfm_cls_dims)
        self.pfm_count = len(self.pfm_patch_dims)
        self.tissue_state_count = int(tissue_state_count)
        if self.tissue_state_count < 0 or self.tissue_state_count == 1:
            raise ValueError("tissue_state_count must be 0 (disabled) or at least 2")
        if self.tissue_state_count and joint_prediction:
            raise ValueError("Tissue states and P0-conditioned joint prediction are alternative architectures")

        self.spatial_pool = int(spatial_pool)
        self.stem = Stem(
            in_chans=3,
            out_chans=encoder_chans[0],
        )
        self.global_dim = global_dim
        self.spatial_dim = spatial_dim
        self.unet = ConvNeXtUnet(
            encoder_chans=encoder_chans,
            decoder_chans=decoder_chans,
            drop_path=drop_path,
            blocks=2,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            pfm_count=self.pfm_count,
            g_top_k=top_k,
            s_top_k=top_k,
            skip_drop_path=skip_drop_path,
        )

        self.feature_channels = int(decoder_chans[-1])
        propagation = nn.Sequential(
            nn.Conv2d(self.feature_channels, propagation_channels, 3, padding=1),
            nn.GELU(),
        )
        segmentation_head = nn.Conv2d(propagation_channels, 1, 1)
        # 保留初始化抽样顺序，使同 seed 的 MoMu 与旧模型一致；新结构不注册旧头。
        if not self.tissue_state_count:
            self.propagation = propagation
            self.segmentation_head = segmentation_head
        self.affinity_head = nn.Conv2d(self.feature_channels, 2, 1)

        self.global_decomposer = PFMDecomposer(
            pfm_cls_dims, global_dim, spatial=False
        )
        self.spatial_decomposer = PFMDecomposer(
            pfm_patch_dims, spatial_dim, spatial=True
        )
        self.global_classifier = nn.Linear(global_dim, self.pfm_count)
        self.spatial_classifier = nn.Linear(spatial_dim, self.pfm_count)

        if backbone_init == "convnext":
            # 先完成原初始化, 再局部重置; 保持其他分支、后续投影头和训练 RNG 不变。
            # 模型在 CPU 构建, 这里只隔离 CPU 随机数状态。
            with torch.random.fork_rng(devices=[]):
                init_convnext_backbone(self.stem)
                init_convnext_backbone(self.unet)

        self.gene_count = int(gene_count)
        self.joint_prediction = bool(joint_prediction)
        self.detach_phenotype = bool(detach_phenotype)
        self.use_seg_condition = bool(use_seg_condition)
        self.use_gene_feedback = bool(use_gene_feedback)
        if self.joint_prediction and (self.gene_count <= 0 or phenotype_dim <= 0 or molecular_dim <= 0):
            raise ValueError("Joint prediction requires a gene head and positive phenotype/molecular dimensions")
        self.gene_head = None
        if self.tissue_state_count:
            with torch.random.fork_rng(devices=[]):
                self.tissue_state_decoder = TissueStateDecoder(
                    self.feature_channels, self.gene_count, self.tissue_state_count, tissue_state_dim,
                )
        elif self.gene_count:
            # 新分支不改变骨干初始化和后续训练的随机数序列。
            with torch.random.fork_rng(devices=[]):
                gene_channels = self.feature_channels
                if self.joint_prediction:
                    # 两列权重对应非癌/癌原型; 软概率加权得到每个位置的类别条件。
                    self.phenotype_embedding = nn.Conv2d(2, phenotype_dim, 1, bias=False)
                    self.molecular_encoder = nn.Sequential(
                        nn.Conv2d(self.feature_channels + phenotype_dim, molecular_dim, 1),
                        nn.GELU(),
                        nn.Conv2d(molecular_dim, molecular_dim, 3, padding=1),
                        nn.GELU(),
                    )
                    feedback_channels = self.feature_channels + molecular_dim + 2
                    self.refinement_head = nn.Sequential(
                        nn.Conv2d(feedback_channels, molecular_dim, 3, padding=1),
                        nn.GELU(),
                        nn.Conv2d(molecular_dim, 1, 1),
                    )
                    self.refinement_gate = nn.Conv2d(feedback_channels, 1, 1)
                    nn.init.zeros_(self.refinement_head[-1].get_parameter("weight"))
                    nn.init.zeros_(self.refinement_head[-1].get_parameter("bias"))
                    nn.init.zeros_(self.refinement_gate.weight)
                    nn.init.zeros_(self.refinement_gate.get_parameter("bias"))
                    gene_channels = molecular_dim
                self.gene_head = nn.Conv2d(gene_channels, self.gene_count, 1)

    def forward(self, image, patch, cls, available=None, *, dense_expression=True):
        # PFM 分成任务特征 f1 和身份特征 f2, 屏蔽同时作用于两者。
        task_global, identity_global = self.global_decomposer(
            [value.float().mean(dim=(-2, -1)) for value in cls]
        )
        task_spatial, identity_spatial = self.spatial_decomposer(
            [value.float() for value in patch]
        )
        if available is not None:
            task_global, identity_global, task_spatial, identity_spatial = _mask_pfm_features(
                available, task_global, identity_global, task_spatial, identity_spatial
            )

        if image.dtype == torch.uint8:
            image = image.float().mul_(2.0 / 255.0).sub_(1.0)

        # H&E 经条件 U-Net 解码, 输出位于标签网格。
        routing = []
        decoded_fm = self.unet(self.stem(image), task_global, task_spatial, available, routing)


        # 分割、RW 软标签生成和 PFM 解耦共用本次特征。
        output = {
            "affinity_logits": self.affinity_head(decoded_fm),
            "features": decoded_fm,
            **self._decomposition(task_global, identity_global, task_spatial, identity_spatial),
        }
        if self.tissue_state_count:
            output.update(self.tissue_state_decoder(decoded_fm, dense_expression=dense_expression))
            output["propagation_features"] = output["tissue_states"]
            return output
        propagation = self.propagation(decoded_fm)
        output.update(
            logits=self.segmentation_head(propagation),
            propagation_features=propagation,
        )
        if self.gene_head is not None:
            gene_features = decoded_fm
            if self.joint_prediction:
                initial_logits = output["logits"]
                probability = initial_logits.float().sigmoid()
                posterior = torch.cat((1 - probability, probability), dim=1)
                if self.detach_phenotype:
                    posterior = posterior.detach()
                # 关闭条件时使用零输入, 保留同一表达编码器的深度和宽度。
                condition = self.phenotype_embedding(
                    posterior if self.use_seg_condition else torch.zeros_like(posterior)
                )
                gene_features = self.molecular_encoder(torch.cat((decoded_fm, condition), dim=1))
                residual = torch.zeros_like(initial_logits, dtype=torch.float32)
                gate = torch.zeros_like(residual)
                if self.use_gene_feedback:
                    feedback = torch.cat((decoded_fm, gene_features, posterior), dim=1)
                    residual = self.refinement_head(feedback).float()
                    gate = self.refinement_gate(feedback).float().sigmoid()
                output.update(
                    initial_logits=initial_logits,
                    initial_probability=probability,
                    molecular_latent=gene_features,
                    refinement_residual=residual,
                    refinement_gate=gate,
                    logits=initial_logits.float() + gate * residual,
                )
            # 输出在线性相对表达空间; spot 聚合后才 log1p。
            output["gene_expression"] = F.softplus(self.gene_head(gene_features).float())
        return output


    def _decomposition(self, task_global, identity_global, task_spatial, identity_spatial):
        """导出解耦损失所需的两路统计量：f2 的分类 logits 与参与去相关的 f1/f2。

        空间维池化到 spatial_pool² 而不是整图均值：保留粗空间结构，统计量级
        与 global 侧可比（每个 (样本, PFM) 贡献 16 个向量）。
        """
        count = self.pfm_count
        pooled_task_spatial = [
            F.adaptive_avg_pool2d(value.float(), self.spatial_pool)
            .flatten(2)
            .transpose(1, 2)
            for value in task_spatial
        ]
        pooled_identity_spatial = [
            F.adaptive_avg_pool2d(value.float(), self.spatial_pool)
            .flatten(2)
            .transpose(1, 2)
            for value in identity_spatial
        ]
        return {
            "pfm_logits_global": torch.stack(
                [self.global_classifier(value.float()) for value in identity_global]
            ).reshape(-1, count),
            "pfm_logits_spatial": torch.stack(
                [
                    self.spatial_classifier(value.mean(dim=1))
                    for value in pooled_identity_spatial
                ]
            ).reshape(-1, count),
            "pfm_f1_global": torch.stack(task_global).reshape(-1, self.global_dim),
            "pfm_f2_global": torch.stack(identity_global).reshape(-1, self.global_dim),
            "pfm_f1_spatial": torch.stack(pooled_task_spatial).reshape(-1, self.spatial_dim),
            "pfm_f2_spatial": torch.stack(pooled_identity_spatial).reshape(-1, self.spatial_dim),
        }
