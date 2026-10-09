"""传播监督: 隐藏类别校准图, 分子分支可用观测重建或纯图像预测。"""

import torch
from torch import nn
from torch.nn import functional as F

from segmodel.modules.graph_measurements import (
    node_average,
    pool_spot_expression,
    reconstruct_expression,
    select_measurements,
    spot_measurements,
    spot_node_ids,
)


class PropagationSupervision(nn.Module):
    def __init__(
        self, feature_channels, gene_count, spot_radius_px, label_scale=1, *,
        query_fraction=0.25, detach_pseudo_target=True, gene_mode="none",
        expression_transform="log1p", gene_smoothing=1.0, gene_ridge=1e-3,
        gene_steps=85, gene_tolerance=1e-6, shuffle_genes=False, seed=0,
        gene_edge_strength=0.3, gene_edge_beta=1.0, detach_gene_edges=False,
    ):
        super().__init__()
        if not 0 < query_fraction < 1:
            raise ValueError("query_fraction must be between 0 and 1")
        if gene_mode not in {"none", "auxiliary", "propagation", "prediction"}:
            raise ValueError(f"Unknown gene_mode: {gene_mode}")
        if expression_transform not in {"log1p", "identity"}:
            raise ValueError("expression_transform must be log1p or identity")
        if gene_smoothing <= 0 or gene_ridge <= 0 or gene_steps < 1 or gene_tolerance <= 0:
            raise ValueError("Gene smoothing, ridge, steps and tolerance must be positive")
        if not 0 <= gene_edge_strength < 1 or gene_edge_beta <= 0:
            raise ValueError("gene_edge_strength must be in [0, 1), gene_edge_beta must be positive")
        if shuffle_genes and gene_mode != "propagation":
            raise ValueError("shuffle_genes only applies to anchor-expression propagation")
        self.spot_radius_px = float(spot_radius_px)
        self.label_scale = int(label_scale)
        self.query_fraction = float(query_fraction)
        self.detach_pseudo_target = bool(detach_pseudo_target)
        self.gene_mode = gene_mode
        self.expression_transform = expression_transform
        self.gene_smoothing = float(gene_smoothing)
        self.gene_ridge = float(gene_ridge)
        self.gene_steps = int(gene_steps)
        self.gene_tolerance = float(gene_tolerance)
        self.shuffle_genes = bool(shuffle_genes)
        self.seed = int(seed)
        self.gene_edge_strength = float(gene_edge_strength)
        self.gene_edge_beta = float(gene_edge_beta)
        self.detach_gene_edges = bool(detach_gene_edges)
        self.gene_head = None
        if gene_mode == "auxiliary":
            # 添加对照分支不改变训练 crop、dropout 等所用的全局随机数状态。
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed + 1907)
                self.gene_head = nn.Linear(feature_channels, gene_count)

    def split_spots(self, batch, step=0):
        """按中心超像素整组隐藏, 防止 query 被同节点的另一个种子直接钳制。

        尽可能为两类分别保留一个 anchor 节点; mask 的 RNG 独立于训练 RNG。
        spot_query 可显式给定分组, 用于固定诊断和信息隔离测试。
        """
        generator = torch.Generator().manual_seed(self.seed + 1000003 * int(step))
        masks = []
        for index, (spots, labels) in enumerate(zip(batch["spots"], batch["spot_labels"])):
            ids = spot_node_ids(batch["superpixels"][index, 0], spots, self.label_scale)
            eligible = ids > 0
            if "spot_query" in batch:
                query_ids = ids[batch["spot_query"][index].bool() & eligible]
            else:
                groups = torch.unique(ids[eligible]).cpu()
                groups = groups[torch.randperm(len(groups), generator=generator)]
                protected = set()
                for label in (0, 1):
                    candidates = set(ids[eligible & (labels == label)].cpu().tolist())
                    chosen = next((int(group) for group in groups if int(group) in candidates), None)
                    if chosen is not None:
                        protected.add(chosen)
                if len(groups) and not protected:
                    protected.add(int(groups[0]))
                candidates = [int(group) for group in groups if int(group) not in protected]
                count = min(len(candidates), max(1, round(len(groups) * self.query_fraction)))
                query_ids = torch.tensor(candidates[:count], device=ids.device, dtype=ids.dtype)
            # 所有同节点 spot 一并隐藏, 包括同节点未知类别但有表达的 spot。
            masks.append(torch.isin(ids, query_ids) & (ids > 0))
        return masks

    def observations(self, graph, batch, query_masks):
        """构造 batch 块对角 A, 一行对应一个 spot, 保留 query 行作最终检查。"""
        entries, weights = [], []
        centers, labels, expression, query, samples, node_samples = [], [], [], [], [], []
        row_offset = 0
        for sample, nodes, offset in zip(graph["indices"], graph["nodes"], graph["offsets"]):
            grid, spots = batch["superpixels"][sample, 0], batch["spots"][sample]
            matrix = spot_measurements(grid, spots, nodes, self.spot_radius_px, self.label_scale)
            rows, columns = matrix.nonzero(as_tuple=True)
            entries.append(torch.stack((rows + row_offset, columns + offset)))
            weights.append(matrix[rows, columns])
            ids = spot_node_ids(grid, spots, self.label_scale)
            local = torch.searchsorted(nodes, ids.contiguous()).clamp_max(len(nodes) - 1)
            centers.append(torch.where(ids > 0, local + offset, -1))
            labels.append(batch["spot_labels"][sample])
            expression.append(batch["gene_expression"][sample].detach().float())
            query.append(query_masks[sample])
            samples.append(torch.full_like(ids, sample))
            node_samples.append(torch.full_like(nodes, sample))
            row_offset += len(spots)
        matrix = torch.sparse_coo_tensor(
            torch.cat(entries, 1), torch.cat(weights),
            (row_offset, len(graph["probability"])),
        ).coalesce()
        valid = torch.zeros(row_offset, dtype=torch.bool, device=matrix.device)
        valid[matrix.indices()[0]] = True
        return {
            "matrix": matrix, "valid": valid, "centers": torch.cat(centers),
            "labels": torch.cat(labels), "expression": torch.cat(expression),
            "query": torch.cat(query), "samples": torch.cat(samples),
            "node_samples": torch.cat(node_samples),
        }

    def gene_query_mask(self, graph, observed):
        """辅助头与传播分支使用同一组有 anchor 支持的 query。"""
        matrix = observed["matrix"]
        anchor = observed["valid"] & ~observed["query"]
        components = graph["components"]
        supported = torch.zeros(int(components.max()) + 1, dtype=torch.bool, device=matrix.device)
        rows, nodes = matrix.indices()
        supported[components[nodes[anchor[rows]]]] = True
        # CUDA 的稀疏 mm 不支持 BF16; 观测矩阵与覆盖判定保持 float32。
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            coverage = torch.sparse.mm(matrix, supported[components, None].float()).squeeze(1)
        return observed["query"] & observed["valid"] & (coverage >= 1 - 1e-6)

    def predict_expression(self, graph, observed, *, step=0):
        """只用 anchor 观测求解; query 表达数值不参与 RHS 或先验计算。"""
        matrix = observed["matrix"]
        anchor = observed["valid"] & ~observed["query"]
        query = self.gene_query_mask(graph, observed)
        if not bool(anchor.any()) or not bool(query.any()):
            return None
        anchor_matrix = select_measurements(matrix, anchor)
        expression = observed["expression"][anchor]
        linear = expression.expm1() if self.expression_transform == "log1p" else expression
        if not bool(torch.isfinite(linear).all()) or bool((linear < 0).any()):
            raise ValueError("Gene supervision requires finite nonnegative linear expression")
        if self.shuffle_genes:
            # 同 crop 内作无固定点置换, 不混合患者, query 真实表达保持原位。
            generator = torch.Generator().manual_seed(self.seed + 1000003 * int(step) + 731)
            sample_ids = observed["samples"][anchor]
            source = torch.arange(len(linear), device=linear.device)
            for sample in sample_ids.unique():
                positions = (sample_ids == sample).nonzero(as_tuple=True)[0]
                order = positions[torch.randperm(len(positions), generator=generator).to(positions.device)]
                source[order] = order.roll(1)
            linear = linear[source]

        # 每个 crop 独立的 anchor 均值先验, 避免 batch 中不同患者互相影响。
        sample_ids = observed["samples"][anchor]
        sample_count = int(observed["node_samples"].max()) + 1
        means = linear.new_zeros(sample_count, linear.shape[1]).index_add(0, sample_ids, linear)
        counts = torch.bincount(sample_ids, minlength=sample_count).clamp_min(1)
        prior = (means / counts[:, None])[observed["node_samples"]]
        with torch.autocast(device_type=linear.device.type, enabled=False):
            field, residual = reconstruct_expression(
                graph["conductance"], graph["edges"], anchor_matrix, linear, prior,
                node_groups=observed["node_samples"],
                smoothing=self.gene_smoothing, ridge=self.gene_ridge,
                steps=self.gene_steps, tolerance=self.gene_tolerance,
            )
            prediction = torch.sparse.mm(matrix.to(dtype=field.dtype), field)

        # 无 anchor 的连通分量只有常数先验, 不能作为成功传播的 query。
        return prediction, query, residual

    def molecular_options(self, output):
        if self.gene_mode != "prediction":
            return {}
        return {
            "molecular_expression": output["gene_expression"],
            "molecular_strength": self.gene_edge_strength,
            "molecular_beta": self.gene_edge_beta,
            "detach_molecular": self.detach_gene_edges,
        }

    def prediction_loss(self, output, batch):
        """真实表达仅作目标, 所有有效 spot 均可监督, 不要求类别种子或图可解。"""
        predictions, targets = [], []
        for field, grid, spots, expression in zip(
            output["gene_expression"], batch["superpixels"][:, 0],
            batch["spots"], batch["gene_expression"],
        ):
            prediction, valid = pool_spot_expression(
                field, grid, spots, self.spot_radius_px, self.label_scale,
            )
            if bool(valid.any()):
                predictions.append(prediction[valid])
                targets.append(expression[valid].detach().float())
        if not predictions:
            return output["gene_expression"].sum() * 0, 0
        with torch.autocast(device_type=predictions[0].device.type, enabled=False):
            prediction = torch.cat(predictions)
            if self.expression_transform == "log1p":
                prediction = prediction.log1p()
            return F.mse_loss(prediction, torch.cat(targets)), len(prediction)

    def forward(self, propagation, output, batch, *, step=0):
        zero = output["features"].float().sum() * 0
        losses = {name: zero for name in (
            "propagation_query", "gene_auxiliary", "gene_propagation", "gene_prediction",
        )}
        diagnostics = {"query_count": 0.0, "label_queries": 0.0, "gene_queries": 0.0, "gene_spots": 0.0}
        if self.gene_mode == "prediction":
            losses["gene_prediction"], diagnostics["gene_spots"] = self.prediction_loss(output, batch)
        query_masks = self.split_spots(batch, step)
        anchor_labels = [
            labels.masked_fill(query, 255)
            for labels, query in zip(batch["spot_labels"], query_masks)
        ]
        propagated, graph = propagation(
            batch["superpixels"].long(), output["features"], batch["image"],
            anchor_labels, batch["spots"], return_graph=True,
            **self.molecular_options(output),
        )
        if graph is None:
            return propagated, losses, diagnostics
        if self.gene_mode == "prediction":
            molecular = graph["molecular_modulation"].detach()
            diagnostics.update({
                "molecular_mean": molecular.mean(), "molecular_std": molecular.std(unbiased=False),
                "molecular_min": molecular.min(), "molecular_max": molecular.max(),
            })
        observed = self.observations(graph, batch, query_masks)
        centers = observed["centers"].clamp_min(0)
        usable = (
            observed["query"] & (observed["centers"] >= 0)
            & (observed["labels"] != 255) & graph["solvable"][centers]
        )
        diagnostics["query_count"] = float(observed["query"].sum())
        diagnostics["label_queries"] = float(usable.sum())
        components = graph["components"]
        counts = [torch.zeros(int(components.max()) + 1, device=centers.device) for _ in (0, 1)]
        for label, count in enumerate(counts):
            count.index_add_(0, components, (graph["seed_labels"] == label).float())
        mixed = (counts[0] > 0) & (counts[1] > 0)
        diagnostics["label_mixed_queries"] = float((usable & mixed[components[centers]]).sum())
        if bool(usable.any()):
            # RW 已输出概率; CUDA autocast 禁止 BCE, 即使输入已转 float32。
            with torch.autocast(device_type=centers.device.type, enabled=False):
                probability = graph["probability"][centers[usable]].float().clamp(1e-6, 1 - 1e-6)
                target = observed["labels"][usable].float()
                losses["propagation_query"] = F.binary_cross_entropy(probability, target)
            diagnostics["label_query_accuracy"] = float(((probability >= 0.5) == target).float().mean())

        prediction = None
        gene_query = self.gene_query_mask(graph, observed) if self.gene_mode in {"auxiliary", "propagation"} else observed["query"]
        if self.gene_mode == "auxiliary" and bool(gene_query.any()):
            features = torch.cat([
                node_average(output["features"][sample], batch["superpixels"][sample, 0], nodes)
                for sample, nodes in zip(graph["indices"], graph["nodes"])
            ])
            with torch.autocast(device_type=features.device.type, enabled=False):
                field = F.softplus(self.gene_head(features))
                prediction = torch.sparse.mm(observed["matrix"], field)
        elif self.gene_mode == "propagation":
            reconstructed = self.predict_expression(graph, observed, step=step)
            if reconstructed is not None:
                prediction, gene_query, residual = reconstructed
                diagnostics["gene_pcg_relative_residual"] = residual.detach()
        if prediction is not None and bool(gene_query.any()):
            prediction = prediction[gene_query]
            if self.expression_transform == "log1p":
                prediction = prediction.log1p()
            key = "gene_auxiliary" if self.gene_mode == "auxiliary" else "gene_propagation"
            losses[key] = F.mse_loss(prediction, observed["expression"][gene_query])
            diagnostics["gene_queries"] = float(gene_query.sum())
        return propagated, losses, diagnostics
