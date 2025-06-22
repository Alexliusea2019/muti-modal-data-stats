#!/usr/bin/env python3
import argparse
import os
import json
import wandb
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, avg, count, floor

# Environment settings
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("HF_DATASETS_OFFLINE", "0"))
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", os.environ.get("HUGGINGFACE_HUB_OFFLINE", "0"))

# Spark settings
env = os.environ
env.setdefault("SPARK_HOME", "/opt/spark")
env.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
env.setdefault("PYSPARK_PYTHON", "python3")

def get_spark(app_name="COCOAnalysis"):
    return SparkSession.builder.appName(app_name).getOrCreate()


def save_coco_to_parquet(output_path="./coco_parquet", max_rows=None):
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "coco_train.parquet")
    # Offline mode: load Karpathy JSON
    if os.getenv("HF_DATASETS_OFFLINE") == "1":
        base = os.path.expanduser("~/.cache/huggingface/modules/datasets_modules/datasets")
        module_dir = next((d for d in os.listdir(base) if d.startswith("HuggingFaceM4--COCO")), None)
        if not module_dir:
            raise RuntimeError("COCO module directory not found under ~/.cache/huggingface/modules/datasets_modules/datasets")
        module_dir = os.path.join(base, module_dir)
        ann_path = os.path.join(module_dir, "karpathy", "dataset_coco.json")
        if not os.path.isfile(ann_path):
            raise RuntimeError(f"Annotation file missing: {ann_path}")
        coco = json.load(open(ann_path, 'r', encoding='utf-8'))
        records = []
        # Iterate each image entry and its sentences
        for img in coco.get("images", []):
            img_id = img.get("imgid") or img.get("id") or img.get("image_id")
            for sent in img.get("sentences", []):
                if max_rows and len(records) >= max_rows:
                    break
                caption = sent.get("raw") or sent.get("caption")
                records.append({
                    "image_id": img_id,
                    "caption": caption,
                    "width": None,
                    "height": None
                })
            if max_rows and len(records) >= max_rows:
                break
        df = pd.DataFrame(records)
    else:
        # Online mode: streaming loader
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceM4/COCO", split="train", streaming=True, trust_remote_code=True
        )
        records = []
        for i, row in enumerate(ds):
            if max_rows and i >= max_rows:
                break
            img = row.get("image", {})
            records.append({
                "image_id": row.get("image_id"),
                "caption": row.get("caption"),
                "width": img.get("width"),
                "height": img.get("height")
            })
        df = pd.DataFrame(records)
    # Save to Parquet
    df.to_parquet(out_file, index=False)
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None, help="Limit the number of rows to load from COCO")
    parser.add_argument("--parquet_path", default="./coco_parquet/coco_train.parquet")
    parser.add_argument("--project", default="wandb_coco")
    parser.add_argument("--run_name", default="coco_analysis")
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.run_name)
    run = wandb.run

    spark = get_spark()

    if not os.path.exists(args.parquet_path):
        args.parquet_path = save_coco_to_parquet(os.path.dirname(args.parquet_path), max_rows=args.max_rows)

    df = spark.read.parquet(args.parquet_path)

    # Add caption_length column
    df = df.withColumn("caption_length", length(col("caption")))

    # Group by image_id to get num_captions and avg_caption_length
    grouped = df.groupBy("image_id").agg(
        count("caption").alias("num_captions"),
        avg("caption_length").alias("avg_caption_length")
    ).toPandas()

    # Log per-image stats
    for i, row in grouped.iterrows():
        wandb.log({
            "avg_caption_length": row["avg_caption_length"],
            "num_captions": row["num_captions"]
        }, step=i)

    # Caption length histogram binning
    bin_df = df.withColumn("length_bin", floor(col("caption_length") / 5) * 5)
    bin_counts = bin_df.groupBy("length_bin").count().orderBy("length_bin").toPandas()
    wandb.log({"caption_length_distribution": wandb.Table(dataframe=bin_counts)}, step=len(grouped))

    # Log image size stats
    image_stats = df.select(avg("width").alias("avg_width"), avg("height").alias("avg_height")).first()
    wandb.log({
        "avg_image_width": image_stats["avg_width"],
        "avg_image_height": image_stats["avg_height"]
    }, step=len(grouped))

    run.finish()
    spark.stop()


if __name__ == "__main__":
    main()

## https://wandb.ai/alexliusea2019-purdue-university/wandb_coco/runs/7q6deh7c?nw=nwuseralexliusea2019S