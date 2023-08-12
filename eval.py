
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import vit as models_vit

from engine_semi import train_one_epoch, evaluate
from util.mixup import get_mixup_func





def get_args_parser():
    parser = argparse.ArgumentParser('Semi-ViT training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--ema_momentum_decay', action='store_true', default=False,
                        help="decay the EMA model momentum")

    # Optimizer parameters
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from MAE pretrained checkpoint')
    parser.add_argument('--super_finetune', default='', help='finetune from supervised pretrained checkpoint')
    parser.add_argument('--super_finetune_ema', action='store_true')
    parser.set_defaults(super_finetune_ema=True)
    parser.add_argument('--no_super_finetune_ema', action='store_false', dest='super_finetune_ema')
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--use_fixed_pos_emb', action='store_true', help='Use sincos position embedding')

    # Dataset parameters
    parser.add_argument('--data_path', default='~/data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                        help='path to train trainindex_x (default: None)')
    parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                        help='path to train trainindex_u (default: None)')
    parser.add_argument('--anno_percent', type=float, default=0.1, help='number of labeled data')

    parser.add_argument('--output_dir', default='/home/renge/Pycharm_Projects/semi-vit/logs/eval',
                        help='path where to save, empty for no saving')
    parser.add_argument('--comment', default='semi-supervised',
                        help='comment')
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--log_dir', default='/home/renge/Pycharm_Projects/semi-vit/logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--tsne', action='store_true',
                        help='tsne')
    parser.add_argument('--no_cuda', action='store_true',
                        help='no_cuda')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--print_freq', default=40, type=int, help='print frequency')

    # Semi-ViT configs:
    parser.add_argument('--ema_teacher', action='store_true', default=False)
    parser.add_argument('--mu', default=5, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda_u', default=5, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--burnin_epochs', default=0, type=int)
    parser.add_argument('--weak_aa', action='store_true', default=False,
                        help="using AutoAugment for weak augmentations")
    parser.add_argument('--weak_no_aug', action='store_true', default=False,
                        help="using disable any augmentation for weak augmentations")
    parser.add_argument('--strong_no_re', action='store_true', default=False,
                        help="disable random erasing for strong augmentations")
    parser.add_argument('--pseudo_mixup', action='store_true', default=False,
                        help="using mixup for unlabeled data")
    parser.add_argument('--disable_x_drop_path', action='store_true', default=False,
                        help="disable drop_path for labeled data")
    parser.add_argument('--disable_x_mixup', action='store_true', default=False,
                        help="disable mixup for labeled data")
    parser.add_argument('--mixup_func', type=str, default='Mixup',
                        help='How to apply mixup/cutmix params. Per "Mixup"')
    parser.add_argument('--pseudo_mixup_func', type=str, default='Mixup',
                        help='How to apply mixup/cutmix params. Per "Mixup", or "ProbPseudoMixup"')

    return parser


def auto_select_gpu(mem_bound=1000, utility_bound=30, gpus=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                    num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print("CUDA_VISIBLE_DEVCIES in os.environ has been set.")
        # sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5):  # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert (len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus
def main(args):
    if not args.no_cuda:
        args.gpu=auto_select_gpu(
        num_gpu=1,
        selected_gpus=None)
    args.distributed = False
    args.print_freq = round(args.print_freq / (args.batch_size * args.mu * misc.get_world_size() / 1024.0))
    if args.color_jitter <= 0:
        args.color_jitter = None
    if args.ema_teacher:
        assert args.model_ema, "EMA teacher requires to enable model_ema"

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device) if not args.no_cuda else torch.device('cpu')

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset_val = build_dataset(False, args=args)
    # sampler_train_x = torch.utils.data.RandomSampler(dataset_train_x)
    # sampler_train_u = torch.utils.data.RandomSampler(dataset_train_u)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_val = torch.utils.data.RandomSampler(dataset_val)
    log_writer = None

    # data_loader_train_x = torch.utils.data.DataLoader(
    #     dataset_train_x, sampler=sampler_train_x,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    # data_loader_train_u = torch.utils.data.DataLoader(
    #     dataset_train_u, sampler=sampler_train_u,
    #     batch_size=args.batch_size * args.mu,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.mu * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn, pseudo_mixup_fn = None, None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_func = get_mixup_func(args.mixup_func)
        mixup_fn = mixup_func(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        print("mixup_fn: {}".format(mixup_fn))
        if args.pseudo_mixup:
            pseudo_mixup_func = get_mixup_func(args.pseudo_mixup_func)
            pseudo_mixup_fn = pseudo_mixup_func(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
            print("pseudo_mixup_fn: {}".format(pseudo_mixup_fn))

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        global_pool=args.global_pool,
        use_fixed_pos_emb=args.use_fixed_pos_emb,
        init_scale=args.init_scale,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load MAE pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
    elif args.super_finetune:
        checkpoint = torch.load(args.super_finetune, map_location='cpu')
        print("Load supervised pre-trained checkpoint from: %s" % args.super_finetune)
        checkpoint_model = checkpoint['model']
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print(msg)

    if not args.no_cuda:model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrmodel_emaapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu or args.no_cuda else '',
            resume='')
        model_ema.base_decay = model_ema.decay
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
        if args.super_finetune and 'model_ema' in checkpoint and args.super_finetune_ema:
            print("Loading EMA model from supervised pre-trained checkpoint: %s" % args.super_finetune)
            checkpoint_ema = checkpoint['model_ema']
            msg = model_ema.ema.load_state_dict(checkpoint_ema, strict=True)
            print(msg)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_tensors = sum(1 for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %f' % (n_parameters / 1.e6))
    print('number of learnable param tensors: %d/%d' % (n_parameters_tensors, len(list(model.parameters()))))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective labeled batch size: %d" % eff_batch_size)
    print("effective unlabeled batch size: %d" % (eff_batch_size * args.mu))
    # print("number of labeled training examples = %d" % len(dataset_train_x))
    # print("number of unlabeled training examples = %d" % len(dataset_train_u))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    opt_args = dict(lr=args.lr)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    optimizer = torch.optim.AdamW(param_groups, **opt_args)
    print(f"optimizer: {optimizer}")
    loss_scaler = NativeScaler()

    if mixup_fn is not None and not args.disable_x_mixup:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if model_ema is not None:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, args)
            print(f"Accuracy of the EMA network on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.now_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        args.output_dir = os.path.join(args.output_dir, f'{args.now_time}---{args.comment}')
        args.log_dir = os.path.join(args.output_dir, 'log')
        # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
