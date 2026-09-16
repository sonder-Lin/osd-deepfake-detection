
import csv
import os
import re

import numpy as np
import torch

from core import utils
from data.build import (
    _is_csv_path,
    _parse_category_from_filename,
    list_csv_files,
    list_subfolders,
)
from data.datasets import CSVDataset, GenerativeImageDataset
from training.engine import evaluate
from training.runtime import create_model_and_ema, setup_for_training

def _save_per_group_predictions(args, dataset_val, eval_result, csv_filename, csv_path):

    import re


    match = re.match(r'\d+_([^.]+)\.csv', csv_filename)
    demographic = match.group(1) if match else "unknown"


    method = args.model.lower()


    img_paths = eval_result.get('img_paths', [])
    y_pred_proba = eval_result.get('y_pred_proba', [])

    if len(img_paths) == 0 or len(y_pred_proba) == 0:
        print(f"  [Warning] No predictions to save for {csv_filename}")
        return


    per_group_dir = os.path.join(args.output_dir, "per_group")
    os.makedirs(per_group_dir, exist_ok=True)


    base_name = os.path.splitext(csv_filename)[0]
    output_filename = f"{base_name}_out_{method}.csv"
    output_path = os.path.join(per_group_dir, output_filename)


    predictions_data = []
    for i, (img_path, prob) in enumerate(zip(img_paths, y_pred_proba)):

        if hasattr(dataset_val, 'get_sample_info'):
            sample_info = dataset_val.get_sample_info(i)
            label = sample_info['label']
            ismale = sample_info['ismale']
            iswhite = sample_info['iswhite']
            isblack = sample_info['isblack']
        else:

            label = ""
            ismale = ""
            iswhite = ""
            isblack = ""

        predictions_data.append({
            'img_path': img_path,
            'label': label,
            'ismale': ismale,
            'iswhite': iswhite,
            'isblack': isblack,
            'prob': prob,
            'demographic': demographic,
            'method': method,
        })


    fieldnames = ['img_path', 'label', 'ismale', 'iswhite', 'isblack', 'prob', 'demographic', 'method']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions_data)

    print(f"  [Saved] {output_path} ({len(predictions_data)} samples)")

def _merge_per_group_predictions(args):

    per_group_dir = os.path.join(args.output_dir, "per_group")
    if not os.path.exists(per_group_dir):
        return

    method = args.model.lower()


    dataset_name = os.path.basename(args.output_dir.rstrip('/\\'))


    csv_files = sorted([f for f in os.listdir(per_group_dir) if f.endswith('.csv')])
    if not csv_files:
        return


    all_rows = []
    fieldnames = ['img_path', 'label', 'ismale', 'iswhite', 'isblack', 'prob', 'demographic', 'method']

    for csv_file in csv_files:
        csv_path = os.path.join(per_group_dir, csv_file)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)


    merged_path = os.path.join(args.output_dir, f"{dataset_name}_merged_all_{method}.csv")
    with open(merged_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[Merged] {merged_path} ({len(all_rows)} total samples from {len(csv_files)} files)")

def run_eval(args):
    device = setup_for_training(args)

    print("Running in Evaluation-Only Mode...")
    model, model_without_ddp, model_ema, _ = create_model_and_ema(args, device)
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None, model_ema=model_ema)


    if getattr(args, 'use_gt_router', False):
        model_without_ddp.use_gt_router = True
        print("[GT Router] Using ground-truth category labels for routing (oracle mode)")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    FOLDER_NAMES = {"GenImage": ("nature", "ai"), "default": ("0_real", "1_fake")}
    real_folder_name_eval, fake_folder_name_eval = FOLDER_NAMES["GenImage" if "GenImage" in args.eval_data_path else "default"]

    domain2label = None
    if "Mirage-Test" in args.eval_data_path:
        domain2label = {'Human': 0, 'Animal': 1, 'Object': 2, 'Scene': 3, 'Anime': 4}

    elif "GenImage_sd14_classfied" in args.eval_data_path:
        domain2label = {'Human_Animal': 0, 'Object_Scene': 1}


    is_single_csv = _is_csv_path(args.eval_data_path)
    csv_files = list_csv_files(args.eval_data_path) if not is_single_csv else [args.eval_data_path]
    is_csv_eval = is_single_csv or len(csv_files) > 0

    if is_csv_eval:

        vals = [os.path.basename(f) for f in csv_files]
        print(f"CSV evaluation mode: found {len(vals)} CSV files")
    else:
        vals = list_subfolders(args.eval_data_path)


        if "DRCT-2M" in args.eval_data_path and len(vals) == 16:
            vals = ['ldm-text2im-large-256/val2017', 'stable-diffusion-v1-4/val2017', 'stable-diffusion-v1-5/val2017', 'stable-diffusion-2-1/val2017',
            'stable-diffusion-xl-base-1.0/val2017', 'stable-diffusion-xl-refiner-1.0/val2017',
            'sd-turbo/val2017', 'sdxl-turbo/val2017',
            'lcm-lora-sdv1-5/val2017', 'lcm-lora-sdxl/val2017',
            'sd-controlnet-canny/val2017', 'sd21-controlnet-canny/val2017', 'controlnet-canny-sdxl-1.0/val2017',
            'stable-diffusion-inpainting/val2017', 'stable-diffusion-2-inpainting/val2017', 'stable-diffusion-xl-1.0-inpainting-0.1/val2017']
        elif "AIGCDetectionBenchMark" in args.eval_data_path and len(vals) == 17:
            vals = ["progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal", "ADM", "Glide",
            "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong", "DALLE2", "sd_xl"]
        elif "GenImage" in args.eval_data_path and len(vals) == 8:
            vals = ["Midjourney/imagenet_midjourney/val", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val",
                    "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val", "ADM/imagenet_ai_0508_adm/val", "glide/imagenet_glide/val",
                    "wukong/imagenet_ai_0424_wukong/val", "VQDM/imagenet_ai_0419_vqdm/val", "BigGAN/imagenet_ai_0419_biggan/val"]
        elif "Mirage-Test" in args.eval_data_path and len(vals) == 5:
            vals = ['Human', 'Animal', 'Object', 'Scene','Anime']
        elif len(vals) == 0:
            vals = [args.eval_data_path]

    rows = [["{} model testing on...".format(args.resume)],
        ['testset', 'accuracy', "real_accuracy", "fake_accuracy", 'avg precision', 'f1_score', 'fnr']]

    num_testsets = len(vals)


    all_y_true = []
    all_y_pred_proba = []
    all_routing_preds = []
    all_routing_labels = []
    all_category_labels = []
    all_features = []

    for v_id, val in enumerate(vals):

        if is_csv_eval:

            csv_path = csv_files[v_id]
            cat_label = _parse_category_from_filename(csv_path)

            override_cat_label = cat_label if cat_label >= 0 else None
            dataset_val = CSVDataset(csv_path=csv_path, is_train=False, resolution=getattr(args, "input_size", 224), override_category_label=override_cat_label)
        else:
            eval_data_path = os.path.join(args.eval_data_path, val)
            dataset_val = GenerativeImageDataset(root=eval_data_path, is_train=False, category2label=domain2label, real_folder_name=real_folder_name_eval, fake_folder_name=fake_folder_name_eval)

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        eval_result = evaluate(data_loader_val, model, device)
        test_stats = eval_result['metrics']
        acc, real_acc, fake_acc = eval_result['acc'], eval_result['real_acc'], eval_result['fake_acc']
        ap, f1, fnr, routing_acc = eval_result['ap'], eval_result['f1'], eval_result['fnr'], eval_result['routing_acc']

        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        print(f"test dataset is {val} acc: {acc}, real acc: {real_acc}, fake acc: {fake_acc}, ap: {ap}, f1: {f1}, fnr: {fnr}, routing_acc: {routing_acc}")
        print("***********************************")


        if eval_result['y_true'] is not None:
            all_y_true.append(eval_result['y_true'])
            all_y_pred_proba.append(eval_result['y_pred_proba'])
        if eval_result['routing_preds'] is not None:
            all_routing_preds.append(eval_result['routing_preds'])
            all_routing_labels.append(eval_result['routing_labels'])
        if eval_result.get('category_labels') is not None:
            all_category_labels.append(eval_result['category_labels'])
        if eval_result.get('features') is not None:
            all_features.append(eval_result['features'])

        rows.append([val, acc if acc is not None else 'N/A',
                     real_acc if real_acc is not None else 'N/A',
                     fake_acc if fake_acc is not None else 'N/A',
                     ap if ap is not None else 'N/A',
                     f1 if f1 is not None else 'N/A',
                     fnr if fnr is not None else 'N/A'])


        save_predictions = str(getattr(args, 'save_predictions', 'False')).lower() in ('true', '1', 'yes')
        if save_predictions and is_csv_eval:
            _save_per_group_predictions(
                args=args,
                dataset_val=dataset_val,
                eval_result=eval_result,
                csv_filename=val,
                csv_path=csv_path,
            )


    if len(all_y_true) > 0:
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, classification_report, roc_auc_score

        y_true_all = np.concatenate(all_y_true)
        y_pred_proba_all = np.concatenate(all_y_pred_proba)

        pred_labels_all = (y_pred_proba_all > 0.5).astype(int)

        overall_acc = accuracy_score(y_true_all, pred_labels_all)
        overall_ap = average_precision_score(y_true_all, y_pred_proba_all) if len(np.unique(y_true_all)) > 1 else None
        overall_auc = roc_auc_score(y_true_all, y_pred_proba_all) if len(np.unique(y_true_all)) > 1 else None
        overall_f1 = f1_score(y_true_all, pred_labels_all, zero_division=0)

        report = classification_report(y_true_all, pred_labels_all, target_names=['real', 'fake'], output_dict=True, zero_division=0)
        overall_real_acc = report['real']['recall']
        overall_fake_acc = report['fake']['recall']
        overall_fnr = 1.0 - overall_fake_acc


        overall_routing_acc = None
        if len(all_routing_preds) > 0:
            routing_preds_all = np.concatenate(all_routing_preds)
            routing_labels_all = np.concatenate(all_routing_labels)
            valid_mask = routing_labels_all >= 0
            if valid_mask.sum() > 0:
                overall_routing_acc = accuracy_score(routing_labels_all[valid_mask], routing_preds_all[valid_mask])

        overall_acc_str = f'{overall_acc:.4f}'
        overall_real_str = f'{overall_real_acc:.4f}'
        overall_fake_str = f'{overall_fake_acc:.4f}'
        overall_ap_str = f'{overall_ap:.4f}' if overall_ap is not None else 'N/A'
        overall_auc_str = f'{overall_auc:.4f}' if overall_auc is not None else 'N/A'
        overall_f1_str = f'{overall_f1:.4f}'
        overall_fnr_str = f'{overall_fnr:.4f}'
        overall_routing_str = f'{overall_routing_acc:.4f}' if overall_routing_acc is not None else 'N/A'

        rows.append(['Overall', overall_acc_str, overall_real_str, overall_fake_str, overall_ap_str, overall_f1_str, overall_fnr_str])
        print(f"\n{'='*60}")
        print(f"OVERALL ({len(y_true_all)} samples) -> Acc: {overall_acc_str}, Real Acc: {overall_real_str}, Fake Acc: {overall_fake_str}, AP: {overall_ap_str}, AUC: {overall_auc_str}, F1: {overall_f1_str}, FNR: {overall_fnr_str}, Routing Acc: {overall_routing_str}")


        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_all, pred_labels_all)


        tn, fp, fn, tp = cm.ravel()
        total_real = tn + fp
        total_fake = fn + tp
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX (rows=true, cols=pred)")
        print(f"{'='*60}")
        print(f"                 Pred Real    Pred Fake")
        print(f"  True Real      {tn:8d}     {fp:8d}    (Total: {total_real})")
        print(f"  True Fake      {fn:8d}     {tp:8d}    (Total: {total_fake})")
        print(f"{'='*60}")
        print(f"  TN (Real→Real): {tn}  |  FP (Real→Fake): {fp}")
        print(f"  FN (Fake→Real): {fn}  |  TP (Fake→Fake): {tp}")
        print(f"{'='*60}")
        print("***********************************")

    test_dataset_name  = args.eval_data_path.split('/')[-2] + '_' + args.eval_data_path.split('/')[-1]

    csv_name = os.path.join(args.output_dir, f'{os.path.basename(args.resume)}_{test_dataset_name}.csv')
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)


    save_predictions = str(getattr(args, 'save_predictions', 'False')).lower() in ('true', '1', 'yes')
    if save_predictions and is_csv_eval:
        _merge_per_group_predictions(args)


    save_features = str(getattr(args, 'save_features', 'False')).lower() in ('true', '1', 'yes')
    if save_features and len(all_features) > 0:
        features_all = np.concatenate(all_features)
        labels_all = np.concatenate(all_y_true) if len(all_y_true) > 0 else np.array([])
        categories_all = np.concatenate(all_category_labels) if len(all_category_labels) > 0 else np.array([])
        probs_all = np.concatenate(all_y_pred_proba) if len(all_y_pred_proba) > 0 else np.array([])


        unique_categories = np.unique(categories_all)


        group_sizes = {cat: (categories_all == cat).sum() for cat in unique_categories}
        min_group_size = min(group_sizes.values())


        samples_per_group = min(min_group_size, getattr(args, 'tsne_samples_per_group', 500))

        print(f"t-SNE sampling: {samples_per_group} samples per group (min group has {min_group_size})")
        for cat in unique_categories:
            print(f"  Category {cat}: {group_sizes[cat]} -> {samples_per_group}")

        sampled_indices = []
        for cat in unique_categories:
            cat_indices = np.where(categories_all == cat)[0]
            if len(cat_indices) > samples_per_group:
                sampled = np.random.choice(cat_indices, samples_per_group, replace=False)
            else:
                sampled = cat_indices
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        features_path = os.path.join(args.output_dir, 'features_for_tsne.npz')
        np.savez(
            features_path,
            features=features_all[sampled_indices],
            labels=labels_all[sampled_indices],
            category_labels=categories_all[sampled_indices],
            probs=probs_all[sampled_indices],
        )
        print(f"Features saved for t-SNE: {features_path} ({len(sampled_indices)} samples, {samples_per_group} per group, {len(unique_categories)} groups)")

    return
