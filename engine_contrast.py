
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import util.misc as misc
import util.lr_sched as lr_sched

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07,):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2):
        batch_size1 = emb1.size(0)
        batch_size2 = emb2.size(0)
        loss1 = 0
        loss2 = 0
        loss_contrastive = torch.tensor([0.])
        for i in range(batch_size1):
            for j in range(i, batch_size1):
                if i==j: continue
                loss1 += torch.exp(F.cosine_similarity(emb1[i].unsqueeze(0), emb1[(i+j)%batch_size1].unsqueeze(0)) / self.temperature)
        for i in range(batch_size1):
            for j in range(batch_size2):
                loss2 += torch.exp(F.cosine_similarity(emb1[i].unsqueeze(0), emb2[j].unsqueeze(0)) / self.temperature)
        if loss1 != 0 and loss2 != 0:
            loss_contrastive = -1*torch.log(loss1/loss2)
        loss_contrastive.requires_grad_(True)
        return loss_contrastive

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    if model_ema is not None:
        model_ema.ema.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _, masks) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)
        samples_authorized = samples[masks==True].to(device, non_blocking=True)
        samples_unauthorized = samples[masks==False].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model.module.forward_contrast(samples_unauthorized)
            authorized_outputs = model.module.forward_contrast(samples_authorized)
            loss = criterion(outputs, authorized_outputs)

        loss = loss.to(device, non_blocking=True)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        del samples, masks, loss
        torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
