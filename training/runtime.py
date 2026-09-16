
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from core import utils
from core.utils import NativeScalerWithGradNormCount as NativeScaler
from models import get_model, list_models
from training.optim import create_optimizer

def setup_for_training(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    return device

def create_model_and_ema(args, device):

    print(f"Available models: {list_models()}")
    model = get_model(args.model, config=args)

    model.to(device)


    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    print(f"Total Parameters at model initialization:     {total_params / 1e6:.2f} M")
    print(f"Total number of trainable params at model initialization: {trainable_params / 1e6:.2f} M")

    n_parameters = trainable_params
    return model, model_without_ddp, model_ema, n_parameters

def create_optimizer_criterion_scaler(args, model_without_ddp):
    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = create_optimizer(args, model_without_ddp)
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    print("Criterion: %s" % str(criterion))
    return optimizer, criterion, loss_scaler, mixup_fn
