
import os
import re

import torch
from torch.utils.data import ConcatDataset

from core import utils
from data.datasets import GenerativeImageDataset, CSVDataset, SimpleImageFolderDataset

def list_subfolders(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local data path does not exist: {path}")

    subfolders = []
    print(f"Attempting to list subfolders for path: {path}")

    subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Found subdirectories in local path: {sorted(subfolders)}")

    return sorted(subfolders)

def list_csv_files(path: str) -> list:

    if not os.path.isdir(path):
        return []
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
    return sorted(files)

def _is_csv_path(p: str) -> bool:
    return isinstance(p, str) and p.lower().endswith(".csv")

def _collect_category_counts(dataset) -> dict:

    category_counts = {}


    if hasattr(dataset, 'data_list'):
        for item in dataset.data_list:
            cat_int = int(item.get('category_label', -1))
            if cat_int >= 0:
                category_counts[cat_int] = category_counts.get(cat_int, 0) + 1

    elif hasattr(dataset, 'datasets'):
        for sub_ds in dataset.datasets:
            sub_counts = _collect_category_counts(sub_ds)
            for cat, count in sub_counts.items():
                category_counts[cat] = category_counts.get(cat, 0) + count

    elif hasattr(dataset, 'category_labels'):
        for cat in dataset.category_labels:
            cat_int = int(cat)
            if cat_int >= 0:
                category_counts[cat_int] = category_counts.get(cat_int, 0) + 1

    return category_counts

def compute_ipw_weights(dataset, intersec_to_gender_race: dict = None) -> dict:

    if intersec_to_gender_race is None:

        intersec_to_gender_race = {
            0: ('female', 'black'),
            1: ('male', 'black'),
            2: ('male', 'asian'),
            3: ('female', 'asian'),
            4: ('male', 'white'),
            5: ('female', 'white'),
        }


    category_counts = _collect_category_counts(dataset)
    total_samples = sum(category_counts.values())

    if total_samples == 0:
        print("[IPW Weights] Warning: No samples found, returning empty weights")
        return {}


    gender_counts = {'male': 0, 'female': 0}
    race_counts = {'asian': 0, 'white': 0, 'black': 0}

    for cat, count in category_counts.items():
        if cat in intersec_to_gender_race:
            gender, race = intersec_to_gender_race[cat]
            gender_counts[gender] = gender_counts.get(gender, 0) + count
            race_counts[race] = race_counts.get(race, 0) + count


    p_gender = {k: v / total_samples for k, v in gender_counts.items() if v > 0}
    p_race = {k: v / total_samples for k, v in race_counts.items() if v > 0}


    print(f"\n[IPW Weights] Computed from {total_samples} samples")
    print(f"  Gender distribution: {', '.join([f'{k}: P={v:.4f}' for k, v in p_gender.items()])}")
    print(f"  Race distribution: {', '.join([f'{k}: P={v:.4f}' for k, v in p_race.items()])}")


    weights = {}
    for cat, (gender, race) in intersec_to_gender_race.items():
        if gender in p_gender and race in p_race:
            weights[cat] = 1.0 / (p_gender[gender] * p_race[race])
        else:
            weights[cat] = 1.0


    if len(weights) > 0 and total_samples > 0:
        weighted_mean = sum(
            weights.get(cat, 1.0) * category_counts.get(cat, 0)
            for cat in weights
        ) / total_samples
        if weighted_mean > 0:
            weights = {k: v / weighted_mean for k, v in weights.items()}


    print("  Category weights (normalized):")
    for cat, weight in sorted(weights.items()):
        if cat in intersec_to_gender_race:
            gender, race = intersec_to_gender_race[cat]
            print(f"    {cat} ({gender}-{race}): {weight:.4f}")
    print()

    return weights

def _parse_category_from_filename(csv_path: str) -> int:

    filename = os.path.basename(csv_path)
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def create_dataloaders(args, load_train=True, load_val=True, parse_category_from_filename=False):

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    FOLDER_NAMES = {"GenImage": ("nature", "ai"), "default": ("0_real", "1_fake")}

    dataset_train, data_loader_train = None, None
    dataset_val, data_loader_val = None, None

    domain2label = None

    if "Mirage-Train" in args.data_path:
        domain2label = {'Human': 0, 'Animal': 1, 'Object': 2, 'Scene': 3, 'Anime': 4}

    elif "GenImage_sd14_classfied" in args.data_path:
        domain2label = {'Human_Animal': 0, 'Object_Scene': 1}

    if load_train:
        real_folder_name_train, fake_folder_name_train = FOLDER_NAMES["GenImage" if "GenImage" in args.data_path else "default"]

        data_paths = args.data_path.split(",")


        if len(data_paths) > 1:
            list_of_datasets = []
            print("Combining the following training datasets:")
            for data_path in data_paths:
                print(f" - Loading {data_path}")
                data_path = data_path.strip()


                csv_files = list_csv_files(data_path)
                if csv_files:
                    for csv_path in csv_files:
                        cat_label = _parse_category_from_filename(csv_path) if parse_category_from_filename else None
                        list_of_datasets.append(CSVDataset(csv_path=csv_path, is_train=True, resolution=getattr(args, "input_size", 224), override_category_label=cat_label))
                    continue


                if _is_csv_path(data_path):
                    cat_label = _parse_category_from_filename(data_path) if parse_category_from_filename else None
                    list_of_datasets.append(CSVDataset(csv_path=data_path, is_train=True, resolution=getattr(args, "input_size", 224), override_category_label=cat_label))
                else:
                    list_of_datasets.append(GenerativeImageDataset(
                        root=data_path,
                        is_train=True,
                        category2label=domain2label,
                        real_folder_name=real_folder_name_train,
                        fake_folder_name=fake_folder_name_train
                    ))
            dataset_train = ConcatDataset(list_of_datasets)
            print(f"\nTotal combined training samples: {len(dataset_train)}")


        else:
            single_path = args.data_path.strip()

            csv_files = list_csv_files(single_path)
            if csv_files:
                list_of_datasets = []
                print("Combining CSV training datasets from directory:")
                for csv_path in csv_files:
                    print(f" - Loading {csv_path}")
                    cat_label = _parse_category_from_filename(csv_path) if parse_category_from_filename else None
                    list_of_datasets.append(CSVDataset(csv_path=csv_path, is_train=True, resolution=getattr(args, "input_size", 224), override_category_label=cat_label))
                dataset_train = ConcatDataset(list_of_datasets)
            elif _is_csv_path(single_path):
                cat_label = _parse_category_from_filename(single_path) if parse_category_from_filename else None
                dataset_train = CSVDataset(csv_path=single_path, is_train=True, resolution=getattr(args, "input_size", 224), override_category_label=cat_label)
            else:
                trains = list_subfolders(single_path)


                if "GenImage" in single_path and len(trains) == 8:
                    specific_folders = ["Midjourney/imagenet_midjourney/train", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train",
                            "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train", "ADM/imagenet_ai_0508_adm/train", "glide/imagenet_glide/train",
                            "wukong/imagenet_ai_0424_wukong/train", "VQDM/imagenet_ai_0419_vqdm/train", "BigGAN/imagenet_ai_0419_biggan/train"]
                    list_of_datasets = []
                    print("Combining specific GenImage training datasets:")
                    for train_folder in specific_folders:
                        data_path = os.path.join(single_path, train_folder)
                        print(f" - Loading {train_folder}")
                        subset = GenerativeImageDataset(root=data_path, is_train=True, category2label=domain2label, real_folder_name=real_folder_name_train, fake_folder_name=fake_folder_name_train)
                        list_of_datasets.append(subset)
                    dataset_train = ConcatDataset(list_of_datasets)
                    print(f"\nTotal combined training samples: {len(dataset_train)}")


                else:
                    dataset_train = GenerativeImageDataset(root=single_path, is_train=True, category2label=domain2label, real_folder_name=real_folder_name_train, fake_folder_name=fake_folder_name_train)

        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )


    if load_val and not args.disable_eval and args.eval_data_path:
        real_folder_name_eval, fake_folder_name_eval = FOLDER_NAMES["GenImage" if "GenImage" in args.eval_data_path else "default"]
        eval_path = args.eval_data_path.strip()
        if _is_csv_path(eval_path):
            dataset_val = CSVDataset(csv_path=eval_path, is_train=False, resolution=getattr(args, "input_size", 224))
        else:
            eval_csvs = list_csv_files(eval_path)
            if eval_csvs:
                list_of_datasets = []
                print("Combining CSV eval datasets from directory:")
                for csv_path in eval_csvs:
                    cat_label = _parse_category_from_filename(csv_path)
                    print(f" - Loading {csv_path} (category_label={cat_label})")
                    list_of_datasets.append(CSVDataset(csv_path=csv_path, is_train=False, resolution=getattr(args, "input_size", 224), override_category_label=cat_label))
                dataset_val = ConcatDataset(list_of_datasets)
            else:
                dataset_val = GenerativeImageDataset(root=eval_path, is_train=False, category2label=domain2label, real_folder_name=real_folder_name_eval, fake_folder_name=fake_folder_name_eval)

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        eval_batch_size = getattr(args, 'eval_batch_size', None) or args.batch_size
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=eval_batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    return dataset_train, data_loader_train, dataset_val, data_loader_val
