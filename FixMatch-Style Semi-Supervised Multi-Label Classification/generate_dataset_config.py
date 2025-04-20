import os
import json
import pandas as pd
from scipy.io import arff
from utils.arff_parser import parse_arff_file

def infer_label_count(arff_path):
    try:
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f" Error loading {arff_path}: {e}")
        return -1

    label_count = 0
    for col in reversed(df.columns):
        try:
            values = df[col].unique()
            values_set = set(v.decode() if isinstance(v, bytes) else v for v in values)
            if values_set.issubset({0, 1, 0.0, 1.0, '0', '1', b'0', b'1'}):
                label_count += 1
            else:
                break
        except Exception as e:
            break
    return label_count

from utils.arff_parser import parse_arff_file
import pandas as pd

def load_arff_file(path):
    attr_names, rows = parse_arff_file(path)
    df = pd.DataFrame(rows, columns=attr_names)
    return df


def generate_dataset_config(dataset_dir, output_json_path):
    """
    Generate dataset_config.json by scanning datasets folder.
    Automatically detects:
    - If dataset has a single file: `dataset.arff`
    - Or has split files: `dataset-train.arff` + `dataset-test.arff`
    - And infers label count from .arff contents
    """
    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.arff')]
    config = {}

    # group files by prefix (without -train/-test suffix)
    grouped = {}
    for file in dataset_files:
        base = file.replace('-train', '').replace('-test', '').replace('.arff', '')
        grouped.setdefault(base, []).append(file)

    for name, files in grouped.items():
        entry = {}
        if len(files) == 1:
            arff_path = os.path.join(dataset_dir, files[0])
            entry["file"] = arff_path
            entry["split"] = "random"
            entry["label_count"] = infer_label_count(arff_path)
        else:
            for f in files:
                full_path = os.path.join(dataset_dir, f)
                if '-train' in f:
                    entry["train_file"] = full_path
                    entry["label_count"] = infer_label_count(full_path)
                elif '-test' in f:
                    entry["test_file"] = full_path
            entry["split"] = "predefined"
        config[name] = entry

    with open(output_json_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f" Generated config with {len(config)} datasets at: {output_json_path}")
generate_dataset_config("datasets", "config/dataset_config.json")
