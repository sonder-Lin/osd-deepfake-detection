import os
import math
from typing import Iterable, Optional

import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from core import utils
from core.utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score
)
import numpy as np

import time


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None, sample_weights: dict = None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, category_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            category_labels = None

        else:
            category_labels = category_labels.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                if type(model) is DDP:
                    loss = model.module.get_losses(output, targets, category_labels, criterion)
                else:
                    loss = model.get_losses(output, targets, category_labels, criterion)

        else:
            output = model(samples)
            if type(model) is DDP:
                loss = model.module.get_losses(output, targets, category_labels, criterion)
            else:
                loss = model.get_losses(output, targets, category_labels, criterion)


        gating_logits = None
        if isinstance(output, dict):
            gating_logits = output.get("gating_logits", None)
            output = output["cls"]

        loss_dict = None

        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss["overall_loss"]


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None


        routing_acc = None
        if gating_logits is not None and category_labels is not None:
            routing_preds = gating_logits.argmax(dim=1)
            routing_acc = (routing_preds == category_labels).float().mean()

        if loss_dict is not None:
            for key, value in loss_dict.items():
                metric_logger.update(**{key: value.item()})
        else:
            metric_logger.update(loss=loss_value)

        metric_logger.update(class_acc=class_acc)
        if routing_acc is not None:
            metric_logger.update(routing_acc=routing_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            if loss_dict is not None:
                for key, value in loss_dict.items():
                    log_writer.update(**{key: value.item()}, head="loss")
            else:
                log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="acc")
            if routing_acc is not None:
                log_writer.update(routing_acc=routing_acc, head="acc")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    model.eval()

    start_time = time.time()


    img_paths_all = []

    for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[1]
        img_paths_batch = batch[2]

        category_label = batch[3]
        category_label = category_label.to(device, non_blocking=True)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        if isinstance(img_paths_batch, (list, tuple)):
            img_paths_all.extend(img_paths_batch)
        else:
            img_paths_all.append(img_paths_batch)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images, category_labels=category_label)
                if type(model) is DDP:
                    loss = model.module.get_losses(output, target, category_label, criterion)
                else:
                    loss = model.get_losses(output, target, category_label, criterion)

        else:
            output = model(images, category_labels=category_label)
            if type(model) is DDP:
                loss = model.module.get_losses(output, target, category_label, criterion)
            else:
                loss = model.get_losses(output, target, category_label, criterion)


        gating_logits_batch = output.get('gating_logits', None) if isinstance(output, dict) else None
        features_batch = output.get('features', None) if isinstance(output, dict) else None
        output_cls = output["cls"] if isinstance(output, dict) else output

        loss_dict = None

        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss["overall_loss"]

        loss_value = loss.item()


        if index == 0:
            predictions = output_cls
            labels = target
            gating_logits_all = gating_logits_batch
            category_labels_all = category_label
            features_all = features_batch
        else:
            predictions = torch.cat((predictions, output_cls), 0)
            labels = torch.cat((labels, target), 0)
            if gating_logits_batch is not None and gating_logits_all is not None:
                gating_logits_all = torch.cat((gating_logits_all, gating_logits_batch), 0)
            category_labels_all = torch.cat((category_labels_all, category_label), 0)
            if features_batch is not None and features_all is not None:
                features_all = torch.cat((features_all, features_batch), 0)

        torch.cuda.synchronize()

        acc1, acc5 = accuracy(output_cls, target, topk=(1, 2))

        batch_size = images.shape[0]
        if loss_dict is not None:
            for key, value in loss_dict.items():
                metric_logger.update(**{key: value.item()})
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    total_time = time.time() - start_time
    num_samples = len(data_loader.dataset)
    fps = num_samples / total_time

    print(f" Inference Time: {total_time:.2f}s")
    print(f" Throughput (FPS): {fps:.2f} samples/sec")


    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                  losses=metric_logger.loss if loss_dict is None else metric_logger.overall_loss))


    if dist.is_initialized() and utils.get_world_size() > 1:
        output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
        dist.all_gather(output_ddp, predictions)
        labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
        dist.all_gather(labels_ddp, labels)
        output_all = torch.concat(output_ddp, dim=0)
        labels_all = torch.concat(labels_ddp, dim=0)
    else:
        output_all = predictions
        labels_all = labels


    y_true = labels_all.detach().cpu().numpy().astype(int)
    y_pred_proba = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    pred_labels = (y_pred_proba > 0.5).astype(int)

    unique_labels = np.unique(y_true)

    acc = None
    ap = None
    f1 = None
    fnr = None
    real_acc = None
    fake_acc = None


    is_real_missing = 0 not in unique_labels
    is_fake_missing = 1 not in unique_labels

    if is_real_missing and is_fake_missing:
        print("Warning: Test set is empty or contains neither class real nor class fake. All metrics set to None.")

    else:
        acc = accuracy_score(y_true, pred_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Test set only contains one class ({unique_labels[0]}). AP set to None.")
        else:
            ap = average_precision_score(y_true, y_pred_proba)
            f1 = f1_score(y_true, pred_labels, zero_division=0)


        report = classification_report(y_true, pred_labels, labels=[0, 1], target_names=['real', 'fake'], output_dict=True, zero_division=0)

        if not is_real_missing and 'real' in report:
            real_acc = report['real']['recall']

        if not is_fake_missing and 'fake' in report:
            fake_acc = report['fake']['recall']
            fnr = 1.0 - fake_acc


    routing_acc = None
    routing_preds_out = None
    routing_labels_out = None
    if gating_logits_all is not None and category_labels_all is not None:
        routing_labels_out = category_labels_all.cpu().numpy()


        model_to_check = model.module if hasattr(model, 'module') else model
        use_gt_router = getattr(model_to_check, 'use_gt_router', False)

        if use_gt_router:

            routing_preds_out = routing_labels_out.copy()
        else:

            routing_preds_out = gating_logits_all.argmax(dim=-1).cpu().numpy()


        valid_mask = routing_labels_out >= 0
        if valid_mask.sum() > 0:
            routing_acc = accuracy_score(routing_labels_out[valid_mask], routing_preds_out[valid_mask])
            mode_str = "(GT Router)" if use_gt_router else "(Network Prediction)"
            print(f"Routing Accuracy: {routing_acc:.4f} ({valid_mask.sum()} samples with valid labels) {mode_str}")
        else:
            print("Routing Accuracy: N/A (no valid category labels)")
    else:
        print("Routing Accuracy: N/A (gating_logits not available)")


    category_labels_np = category_labels_all.cpu().numpy() if category_labels_all is not None else None


    features_np = features_all.cpu().numpy() if features_all is not None else None

    return {
        'metrics': {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        'acc': acc, 'real_acc': real_acc, 'fake_acc': fake_acc,
        'ap': ap, 'f1': f1, 'fnr': fnr, 'routing_acc': routing_acc,
        'y_true': y_true, 'y_pred_proba': y_pred_proba,
        'routing_preds': routing_preds_out, 'routing_labels': routing_labels_out,
        'category_labels': category_labels_np,
        'img_paths': img_paths_all,
        'features': features_np,
    }
