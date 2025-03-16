import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random
import shutil


def get_split_json_files(data_dir_path):
    split_keys = ['train', 'val', 'test']
    annontation_categories = ['cell', 'table']

    split_json_files = {}

    for annontation_category in annontation_categories:
        for file_path in data_dir_path.glob('*.jsonl'):
            if annontation_category not in file_path.stem:
                continue

            split_json_files.setdefault(annontation_category, {})
            ann_files = split_json_files[annontation_category]

            for split_key in split_keys:
                if split_key not in file_path.stem:
                    continue

                assert split_key not in ann_files, ann_files[split_key]
                ann_files[split_key] = str(file_path)

    print("Parsed annotation file paths:")
    print(json.dumps(split_json_files, indent=4))
    return split_json_files


def select_samples(ann_file_path, num_samples):
    samples = []

    with open(ann_file_path, 'r') as file_stream:
        for line in tqdm(file_stream):
            if line:
                sample = json.loads(line)
                samples.append(sample)

    random.seed(17)
    selected_samples = random.sample(samples, num_samples)
    return selected_samples


def write_data_to_file(ann_file_path, samples):
    with open(ann_file_path, 'w') as file_stream:
        for index, sample in enumerate(samples):
            line = json.dumps(sample)

            if index > 0:
                file_stream.write('\n')

            file_stream.write(line)


def select_data(split_json_files, tiny_dir_path, num_samples_info):
    for annontation_category, ann_files in split_json_files.items():
        for split_key, ann_file_path in ann_files.items():
            print(f"Selecting samples from {ann_file_path}")
            num_samples = num_samples_info[split_key]
            samples = select_samples(ann_file_path, num_samples)

            tiny_ann_file_path = tiny_dir_path / Path(ann_file_path).name
            print(f"Writing selected samples to {tiny_ann_file_path}")
            write_data_to_file(tiny_ann_file_path, samples)


def copy_pdf_files(data_dir_path, tiny_dir_path):
    for ann_file_path in tiny_dir_path.glob('*.jsonl'):
        print(f"Copying image pdf files listed from {ann_file_path}")

        samples = []

        with open(ann_file_path, 'r') as file_stream:
            for line in tqdm(file_stream):
                if line:
                    sample = json.loads(line)
                    samples.append(sample)

        for sample in tqdm(samples):
            src_file_path = data_dir_path / 'pdf' / sample['filename']
            dest_file_path = tiny_dir_path / 'pdf' / sample['filename']
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file_path, dest_file_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    parser.add_argument('--num_train_samples', type=int, default=800,
                        help="Number of train samples.")
    parser.add_argument('--num_val_samples', type=int, default=100,
                        help="Number of train samples.")
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help="Number of train samples.")
    return parser.parse_args()


def main():
    args = get_args()
    data_dir_path = Path(args.data_dir)
    split_json_files = get_split_json_files(data_dir_path)

    tiny_dir_path = Path(args.output_dir)
    tiny_dir_path.mkdir(parents=True, exist_ok=True)

    num_samples_info = dict(train=args.num_train_samples,
                            val=args.num_val_samples,
                            test=args.num_test_samples)

    select_data(split_json_files, tiny_dir_path, num_samples_info)

    copy_pdf_files(data_dir_path, tiny_dir_path)
    print("Completed!")

if __name__ == '__main__':
    main()
