
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from core import utils
from data.build import (
    _is_csv_path,
    compute_ipw_weights,
    create_dataloaders,
    list_csv_files,
    list_subfolders,
)
from data.datasets import SimpleImageFolderDataset
from training.engine import evaluate, train_one_epoch
from training.runtime import (
    create_model_and_ema,
    create_optimizer_criterion_scaler,
    setup_for_training,
)

def run_training_loop(args, model, model_without_ddp, model_ema, n_parameters, device,
                      criterion, optimizer, loss_scaler, mixup_fn,
                      dataset_train, dataset_val, data_loader_train, data_loader_val,
                      sample_weights: dict = None):

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log_writer = None
    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, args=args,
            sample_weights=sample_weights
            )
        if args.output_dir and args.save_ckpt and not getattr(args, 'save_ckpt_last_only', False):
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            if args.dist_eval and hasattr(data_loader_val.sampler, 'set_epoch'):
                data_loader_val.sampler.set_epoch(epoch)
            eval_result = evaluate(data_loader_val, model, device)
            test_stats = eval_result['metrics']
            acc, real_acc, fake_acc = eval_result['acc'], eval_result['real_acc'], eval_result['fake_acc']
            ap, f1, fnr, routing_acc = eval_result['ap'], eval_result['f1'], eval_result['fnr'], eval_result['routing_acc']

            auc = None
            y_true = eval_result.get('y_true')
            y_pred_proba = eval_result.get('y_pred_proba')
            if y_true is not None and y_pred_proba is not None:
                try:
                    from sklearn.metrics import roc_auc_score
                    import numpy as np
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, y_pred_proba)
                except Exception:
                    auc = None

            val_loss = test_stats.get('overall_loss', test_stats.get('loss', float('inf')))
            auc_str = f", AUC: {auc:.4f}" if auc is not None else ""
            print(f"Accuracy of the model on the {len(dataset_val)} validation samples: {test_stats['acc1']:.1f}%, val_loss: {val_loss:.4f}{auc_str}")

            if log_writer is not None:
                for k, v in test_stats.items():
                    if "acc" in k or "loss" in k:
                        log_writer.update(**{f'test_{k}': v}, head="perf", step=epoch)
                if auc is not None:
                    log_writer.update(test_auc=auc, head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'val_real_acc': real_acc,
                        'val_fake_acc': fake_acc,
                        'val_auc': auc,
                        'epoch': epoch,
                        'n_parameters': n_parameters}


            if args.model_ema and args.model_ema_eval:
                eval_result_ema = evaluate(data_loader_val, model, device)
                test_stats_ema = eval_result_ema['metrics']
                acc, real_acc, fake_acc = eval_result_ema['acc'], eval_result_ema['real_acc'], eval_result_ema['fake_acc']
                ap, f1, fnr = eval_result_ema['ap'], eval_result_ema['f1'], eval_result_ema['fnr']
                print(f"Accuracy of the model EMA on {len(dataset_val)} validation samples: {test_stats_ema['acc1']:.1f}%, ap: {ap}")
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.output_dir and args.save_ckpt and getattr(args, 'save_ckpt_last_only', False):
        utils.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch="last", model_ema=model_ema)
        print(f"Final checkpoint saved to {os.path.join(args.output_dir, 'checkpoint-last.pth')}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def load_and_merge_expert_weights(model_without_ddp, config_path):

    print(f"\n--- Starting weight merging using config file: {config_path} ---")

    with open(config_path, 'r') as f:
        config = json.load(f)

    base_dir = config['stage1_base_dir']
    experts_map = config['experts_map']


    merged_state_dict = model_without_ddp.state_dict()


    for expert_idx_str, checkpoint_filename in experts_map.items():
        expert_idx = int(expert_idx_str)
        expert_dir = os.path.join(base_dir, f'expert_{expert_idx}')
        expert_checkpoint_path = os.path.join(expert_dir, checkpoint_filename)

        if not os.path.exists(expert_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for expert {expert_idx} not found at {expert_checkpoint_path}")

        print(f"Loading weights for expert {expert_idx} from: {expert_checkpoint_path}")

        expert_checkpoint = torch.load(expert_checkpoint_path, map_location='cpu')

        if 'model' in expert_checkpoint:
            expert_state_dict = expert_checkpoint['model']
        elif 'model_ema' in expert_checkpoint:
            expert_state_dict = expert_checkpoint['model_ema']
        else:
            expert_state_dict = expert_checkpoint


        expert_pattern = re.compile(f'[USV]_experts\\.{expert_idx}$')
        keys_to_transfer = [key for key in expert_state_dict if expert_pattern.search(key)]

        print(keys_to_transfer)
        if not keys_to_transfer:
            raise ValueError(
                f"Error: No weights found for expert {expert_idx} in checkpoint '{expert_checkpoint_path}'. "
                f"This means the regex pattern could not find any matching weight keys. "
                f"Please check the checkpoint file's contents and the script's logic."
            )

        print(f"  - Transferring {len(keys_to_transfer)} weights for expert {expert_idx}.")
        for key in keys_to_transfer:
            merged_state_dict[key] = expert_state_dict[key]

    model_without_ddp.load_state_dict(merged_state_dict)
    print("--- Weight merging complete. Model is ready for Stage 2. ---\n")

    return model_without_ddp

def run_stage1_hard_sampling(args):
    device = setup_for_training(args)

    data_paths = args.data_path.split(",")


    if len(data_paths) > 1:
        train_domains = []
        for idx, data_path in enumerate(data_paths):
            data_path = data_path.strip()


            csv_files = list_csv_files(data_path)
            if csv_files:


                if idx == len(data_paths) - 1:
                    train_domains.append(data_path)
                else:
                    train_domains.extend(csv_files)
                continue


            if _is_csv_path(data_path):
                train_domains.append(data_path)
                continue

            sub_domains = list_subfolders(data_path)

            if "GenImage" in data_path and len(sub_domains) == 8:
                specific_folders = ["Midjourney/imagenet_midjourney/train", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train",
                        "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train", "ADM/imagenet_ai_0508_adm/train", "glide/imagenet_glide/train",
                        "wukong/imagenet_ai_0424_wukong/train", "VQDM/imagenet_ai_0419_vqdm/train", "BigGAN/imagenet_ai_0419_biggan/train"]

                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])

            elif "Mirage-Train" in data_path and len(sub_domains) == 5:
                specific_folders = ["Human", "Animal", "Object", "Scene", "Anime"]
                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])

            elif "GenImage_sd14_classfied" in data_path and len(sub_domains) == 2:

                specific_folders = ["Human_Animal", "Object_Scene"]
                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])

            else:
                train_domains.append(data_path)

    else:
        single_path = args.data_path.strip()
        csv_files = list_csv_files(single_path)
        if csv_files:
            train_domains = csv_files
        elif _is_csv_path(single_path):
            train_domains = [single_path]
        else:
            train_domains = list_subfolders(single_path)
            train_domains = [os.path.join(single_path, folder) for folder in train_domains]

    assert len(train_domains) == args.num_experts, "Number of experts must match number of data domains."

    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)

    print(f"Starting Stage 1: Training {args.num_experts} experts sequentially...")


    initial_head_state_dict = deepcopy(model_without_ddp.head.state_dict())

    data_path_copy = args.data_path
    output_dir_copy = args.output_dir
    log_dir_copy = args.log_dir

    for expert_idx, domain_name in enumerate(train_domains):
        domain_name = train_domains[expert_idx]

        print(f"\n{'='*25} PREPARING EXPERT {expert_idx} | DOMAIN: {domain_name} {'='*25}")


        expert_output_dir = os.path.join(output_dir_copy, f"expert_{expert_idx}")
        checkpoint_path_final = os.path.join(expert_output_dir, "checkpoint-last.pth")

        if os.path.exists(checkpoint_path_final):
            print(f"[SKIP] Expert {expert_idx} checkpoint already exists: {checkpoint_path_final}")
            print(f"[SKIP] Skipping training for Expert {expert_idx}.")
            continue

        model_without_ddp.head.load_state_dict(initial_head_state_dict)
        print(f"Classification head has been reset to its initial state for expert {expert_idx}.")

        if hasattr(model_without_ddp, 'set_training_mode'):
            model_without_ddp.set_training_mode('hard_sampling', expert_idx)

        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print(f"\nNumber of trainable parameters for Expert {expert_idx}: {n_parameters}")

        args.output_dir = expert_output_dir
        args.log_dir = os.path.join(log_dir_copy, f"expert_{expert_idx}")


        is_artifact_expert = expert_idx == args.num_experts - 1
        use_artifact_aug = is_artifact_expert and getattr(args, 'artifact_aug', False)

        if use_artifact_aug:

            print(f"[Artifact Aug] Enabled for expert {expert_idx}")
            print(f"[Artifact Aug] Dir: {args.artifact_aug_dir}")

            optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)

            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            for epoch in range(args.start_epoch, args.epochs):

                epoch_num = epoch % 10
                epoch_dir = os.path.join(args.artifact_aug_dir, f"epoch_{epoch_num:03d}")
                print(f"\n[Artifact Aug] Epoch {epoch}: Loading images from {epoch_dir}")

                dataset_train = SimpleImageFolderDataset(folder_path=epoch_dir, is_train=True)


                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed + epoch)
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=args.pin_mem, drop_last=True,
                    persistent_workers=True if args.num_workers > 0 else False,
                    prefetch_factor=4 if args.num_workers > 0 else None,
                )

                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)


                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, model_ema, mixup_fn,
                    log_writer=None, args=args
                )

            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="last", model_ema=model_ema)
                print(f"Final checkpoint saved to {os.path.join(args.output_dir, 'checkpoint-last.pth')}")


            print(f"\n===== Finished training Artifact Expert {expert_idx} with augmented data =====")
            continue
        else:

            args.data_path = domain_name
            dataset_train, _, _, _ = create_dataloaders(args, load_train=True, load_val=False)
        optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)


        if hasattr(dataset_train, 'data_list'):
            n_real = sum(1 for s in dataset_train.data_list if s.get('label', s.get('target', -1)) == 0)
            n_fake = sum(1 for s in dataset_train.data_list if s.get('label', s.get('target', -1)) == 1)
        else:

            n_real, n_fake = 0, 0
            for ds in getattr(dataset_train, 'datasets', [dataset_train]):
                if hasattr(ds, 'data_list'):
                    n_real += sum(1 for s in ds.data_list if s.get('label', s.get('target', -1)) == 0)
                    n_fake += sum(1 for s in ds.data_list if s.get('label', s.get('target', -1)) == 1)

        n_total = n_real + n_fake
        if n_real > 0 and n_fake > 0:
            w_real = n_total / (2 * n_real)
            w_fake = n_total / (2 * n_fake)


            sample_weights = []
            if hasattr(dataset_train, 'data_list'):
                for s in dataset_train.data_list:
                    label = s.get('label', s.get('target', 0))
                    sample_weights.append(w_real if label == 0 else w_fake)
            else:
                for ds in getattr(dataset_train, 'datasets', [dataset_train]):
                    if hasattr(ds, 'data_list'):
                        for s in ds.data_list:
                            label = s.get('label', s.get('target', 0))
                            sample_weights.append(w_real if label == 0 else w_fake)


            from torch.utils.data import WeightedRandomSampler
            sampler_train = WeightedRandomSampler(sample_weights, num_samples=len(dataset_train), replacement=True)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=args.pin_mem, drop_last=True,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=4 if args.num_workers > 0 else None,
            )
            print(f"[Expert {expert_idx}] Weighted sampling: real={n_real} (w={w_real:.4f}), fake={n_fake} (w={w_fake:.4f})")
        else:

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, shuffle=True,
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=args.pin_mem, drop_last=True,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=4 if args.num_workers > 0 else None,
            )
            print(f"[Expert {expert_idx}] Warning: Cannot compute sample weights (real={n_real}, fake={n_fake}), using uniform sampling")


        run_training_loop(
            args, model, model_without_ddp, model_ema, n_parameters, device,
            criterion, optimizer, loss_scaler, mixup_fn,
            dataset_train, None, data_loader_train, None
        )
        print(f"\n===== Finished training Expert {expert_idx} for Domain: {domain_name} =====")


    args.data_path = data_path_copy
    args.output_dir = output_dir_copy
    args.log_dir = log_dir_copy

    print("\n--- Stage 1 finished. ---")

def run_stage2_head_finetune(args):


    device = setup_for_training(args)

    args.output_dir = os.path.join(args.output_dir, "head_finetune")
    args.log_dir = os.path.join(args.log_dir, "head_finetune")


    original_data_path = args.data_path
    data_paths = args.data_path.split(",")
    if len(data_paths) > 1:
        args.data_path = data_paths[0].strip()
        print(f"Stage 2: Using only the first data path: {args.data_path}")

    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)


    if hasattr(model_without_ddp, 'set_training_mode'):
        model_without_ddp.set_training_mode('head_finetune')


    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)


    else:
        assert args.moe_config_path, "For Stage 2, a merge config JSON file must be provided via --moe_config_path."
        model_without_ddp = load_and_merge_expert_weights(model_without_ddp, args.moe_config_path)

    optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)

    print(f"Starting Stage 2: Finetuning Head with zero-shot routing ({args.num_experts} experts)...")

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Number of trainable parameters for Head Finetune: {n_parameters}")


    dataset_train, _, _, _ = create_dataloaders(
        args, load_train=True, load_val=False, parse_category_from_filename=True
    )


    ipw_weights = compute_ipw_weights(dataset_train)


    sample_weights = []
    if hasattr(dataset_train, 'datasets'):
        for ds in dataset_train.datasets:
            if hasattr(ds, 'data_list'):
                for s in ds.data_list:
                    cat = s.get('category_label', -1)
                    sample_weights.append(ipw_weights.get(cat, 1.0))
    elif hasattr(dataset_train, 'data_list'):
        for s in dataset_train.data_list:
            cat = s.get('category_label', -1)
            sample_weights.append(ipw_weights.get(cat, 1.0))
    else:

        sample_weights = [1.0] * len(dataset_train)


    from torch.utils.data import WeightedRandomSampler
    sampler_train = WeightedRandomSampler(sample_weights, num_samples=len(dataset_train), replacement=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"[Stage 2] Using WeightedRandomSampler with IPW weights")

    run_training_loop(
        args, model, model_without_ddp, model_ema, n_parameters, device,
        criterion, optimizer, loss_scaler, mixup_fn,
        dataset_train, None, data_loader_train, None,
        sample_weights=None
        )

    print("\n--- Stage 2 finished. ---")

def run_standard_training(args):

    device = setup_for_training(args)


    print("\n--- Running in Standard Training Mode. ---")
    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)
    optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)


    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)


    elif args.resume:
        utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                              optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    dataset_train, data_loader_train, _, _ = create_dataloaders(args, load_train=True, load_val=False)
    run_training_loop(
        args, model, model_without_ddp, model_ema, n_parameters, device,
        criterion, optimizer, loss_scaler, mixup_fn,
        dataset_train, None, data_loader_train, None,
        )
    print("\n--- Standard training finished. ---")
