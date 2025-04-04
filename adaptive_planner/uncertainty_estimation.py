from __future__ import annotations

from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn

from ultralytics.utils.metrics import bbox_iou


class BaseUncertaintyEstimator(ABC, nn.Module):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()

        self.iou_threshold = iou_threshold

    def forward(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        return self.compute_uncertainty(pred, dropout_preds, dropout_cls_confs)

    @abstractmethod
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        pass


class UncertaintyProduct(nn.Module):
    def __init__(self, modules: list[type[BaseUncertaintyEstimator]], iou_threshold: float = 0.5) -> None:
        super().__init__()

        self.uncertainty_estimators = [m(iou_threshold=iou_threshold) for m in modules]

    def __str__(self) -> str:
        return f"Product({','.join([type(m).__name__ for m in self.uncertainty_estimators])})"

    def forward(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        uncertainty = th.ones(pred.size(0), device=pred.device, dtype=th.float32)
        for m in self.uncertainty_estimators:
            uncertainty *= m(pred, dropout_preds, dropout_cls_confs)
        return uncertainty


class UncertaintyEstimatorOCC(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        bbox_counts = th.zeros(pred.size(0), device=pred.device, dtype=th.float32)
        for i in range(pred.size(0)):
            for dp in dropout_preds:
                for j in range(dp.size(0)):
                    iou = bbox_iou(pred[i, :4], dp[j, :4])

                    if iou > self.iou_threshold:
                        bbox_counts[i] += 1
                        break

        return bbox_counts / len(dropout_preds)


class UncertaintyEstimatorCLS(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        assert len(dropout_preds) == len(dropout_cls_confs)

        # Calculate the maximum possible entropy when all class labels have same probability
        n_classes = dropout_cls_confs[0].size(1)
        max_entropy = -n_classes * (1 / n_classes * th.log(th.tensor([1 / n_classes], device=pred.device, dtype=th.float32)))

        semantic_uncertainty = th.zeros(pred.size(0), device=pred.device, dtype=th.float32)
        for i in range(pred.size(0)):
            cls_certainties = []
            for dp, cls_conf in zip(dropout_preds, dropout_cls_confs):
                for j in range(dp.size(0)):
                    iou = bbox_iou(pred[i, :4], dp[j, :4])

                    if iou > self.iou_threshold:
                        cls_certainties.append(cls_conf[j, :])
                        break

            # If there is no match, make sure to return a zero
            if len(cls_certainties) == 0:
                cls_certainties = [th.zeros((n_classes,), device=pred.device, dtype=th.float32)]

            cls_certainty_tensor = th.vstack(cls_certainties)
            entropy = -(cls_certainty_tensor * th.log(cls_certainty_tensor)).sum(dim=1, keepdim=True)
            semantic_uncertainty[i] = th.mean(1 - entropy / max_entropy)

        return semantic_uncertainty


class UncertaintyEstimatorCLS2(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        assert len(dropout_preds) == len(dropout_cls_confs)

        semantic_uncertainty = th.zeros(pred.size(0), device=pred.device, dtype=th.float32)
        for i in range(pred.size(0)):
            cls_certainties = []
            for dp, cls_conf in zip(dropout_preds, dropout_cls_confs):
                for j in range(dp.size(0)):
                    iou = bbox_iou(pred[i, :4], dp[j, :4])

                    if iou > self.iou_threshold:
                        cls_certainties.append(cls_conf[j, :])
                        break

            # If there is no match, make sure to return a zero
            if len(cls_certainties) == 0:
                cls_certainties = [th.zeros(pred.size(1), device=pred.device, dtype=th.float32)]

            cls_certainty_tensor = th.vstack(cls_certainties)
            entropy = -(cls_certainty_tensor * th.log(cls_certainty_tensor)).sum(dim=0, keepdim=True)
            semantic_uncertainty[i] = (1 - th.softmax(entropy, dim=1)).max()

        return semantic_uncertainty


class UncertaintyEstimatorIOU(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        uncertainty = th.empty(pred.size(0), device=pred.device, dtype=th.float32)
        for i in range(pred.size(0)):
            iou_values = []
            for dp in dropout_preds:
                for j in range(dp.size(0)):
                    iou = bbox_iou(pred[i, :4], dp[j, :4])

                    if iou > self.iou_threshold:
                        iou_values.append(iou)
                        break

            # If there is no match, make sure to return a zero
            if len(iou_values) == 0:
                iou_values = [th.zeros((1,), device=pred.device, dtype=th.float32)]

            uncertainty[i] = th.cat(iou_values).mean()

        # Rescale the uncerties since the iou_threshold is the minimum value
        return (uncertainty - self.iou_threshold) / (1.0 - self.iou_threshold)


class UncertaintyEstimatorYOLO(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        return pred[:, 4]


class UncertaintyEstimatorYOLOMean(BaseUncertaintyEstimator):
    def compute_uncertainty(self, pred: th.Tensor, dropout_preds: list[th.Tensor], dropout_cls_confs: list[th.Tensor]) -> th.Tensor:
        uncertainty = th.empty(pred.size(0), device=pred.device, dtype=th.float32)
        for i in range(pred.size(0)):
            conf_values = []
            for dp in dropout_preds:
                for j in range(dp.size(0)):
                    iou = bbox_iou(pred[i, :4], dp[j, :4])

                    if iou > self.iou_threshold:
                        conf_values.append(dp[j, 4].unsqueeze(0))
                        break

            # If there is no match, make sure to return a zero
            if len(conf_values) == 0:
                conf_values = [th.zeros((1,), device=pred.device, dtype=th.float32)]

            uncertainty[i] = th.cat(conf_values).mean()

        return uncertainty
