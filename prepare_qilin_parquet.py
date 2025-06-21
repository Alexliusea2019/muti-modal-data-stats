import argparse
import os
import pandas as pd
from datasets import load_dataset

def save_qilin_to_parquet(config_name, output_path="./qilin_parquet"):
    print(f"Loading config: {config_name} from THUIR/Qilin")
    ds = load_dataset("THUIR/Qilin", config_name, split="train")
    records = []
    for row in ds:
        record = row.copy()
        records.append(record)
    df = pd.DataFrame(records)
    os.makedirs(output_path, exist_ok=True)
    parquet_path = os.path.join(output_path, f"qilin_{config_name}.parquet")
    df.to_parquet(parquet_path)
    print(f"Saved parquet to: {parquet_path}")
    return parquet_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True, choices=["search_train", "search_test", "recommendation_train", "recommendation_test"])
    parser.add_argument("--output_path", default="./qilin_parquet")
    args = parser.parse_args()

    save_qilin_to_parquet(args.config_name, args.output_path)

if __name__ == "__main__":
    main()
