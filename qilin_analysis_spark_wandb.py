import argparse
import os
import wandb
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, explode, avg, length
import pandas as pd

def get_spark(app_name="QilinAnalysis"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def save_qilin_to_parquet(config_name, output_path="./qilin_parquet"):
    ds = load_dataset("THUIR/Qilin", config_name, split="train")
    records = []
    for row in ds:
        record = row.copy()
        records.append(record)
    df = pd.DataFrame(records)
    path = os.path.join(output_path, f"qilin_{config_name}.parquet")
    os.makedirs(output_path, exist_ok=True)
    df.to_parquet(path)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True, choices=["search_train", "search_test", "recommendation_train", "recommendation_test"])
    parser.add_argument("--parquet_path", default="./qilin_parquet")
    parser.add_argument("--project", default="wandb_qilin")
    parser.add_argument("--run_name", default="qilin_analysis")
    args = parser.parse_args()

    parquet_file = os.path.join(args.parquet_path, f"qilin_{args.config_name}.parquet")

    if not os.path.exists(parquet_file):
        parquet_file = save_qilin_to_parquet(args.config_name, args.parquet_path)

    wandb.init(project=args.project, name=args.run_name)
    run = wandb.run

    spark = get_spark()
    df = spark.read.parquet(parquet_file)

    if args.config_name.startswith("search"):
        df = df.withColumn("query_length", length(col("query")))
        df = df.withColumn("num_clicks", size(col("dpr_results")))
        df = df.withColumn("num_candidates", size(col("search_result_details_with_idx")))

        grouped = df.groupBy("user_idx").agg(
            avg("query_length").alias("avg_query_length"),
            avg("num_clicks").alias("avg_clicks_per_query"),
            avg("num_candidates").alias("avg_candidates_per_query")
        ).toPandas()

        for i, row in grouped.iterrows():
            wandb.log({
                "avg_query_length": row["avg_query_length"],
                "avg_clicks_per_query": row["avg_clicks_per_query"],
                "avg_candidates_per_query": row["avg_candidates_per_query"]
            }, step=i)

        exploded_clicks = df.select(explode(col("dpr_results")).alias("note_id"))
        top_clicked = exploded_clicks.groupBy("note_id").count().orderBy(col("count").desc()).limit(10).toPandas()
        wandb.log({"top_10_clicked_note_ids": wandb.Table(dataframe=top_clicked)}, step=len(grouped))

    elif args.config_name.startswith("recommendation"):
        df = df.withColumn("rec_list_size", size(col("rec_result_details_with_idx")))
        df = df.withColumn("recent_clicks", size(col("recent_clicked_note_idxs")))

        grouped = df.groupBy("user_idx").agg(
            avg("rec_list_size").alias("avg_rec_list_size"),
            avg("recent_clicks").alias("avg_recent_clicks")
        ).toPandas()

        for i, row in grouped.iterrows():
            wandb.log({
                "avg_rec_list_size": row["avg_rec_list_size"],
                "avg_recent_clicks": row["avg_recent_clicks"]
            }, step=i)

        exploded_recs = df.select(explode(col("rec_result_details_with_idx")).alias("rec_item"))
        top_recs = exploded_recs.groupBy("rec_item").count().orderBy(col("count").desc()).limit(10).toPandas()
        wandb.log({"top_10_recommended_note_idxs": wandb.Table(dataframe=top_recs)}, step=len(grouped))

    run.finish()
    spark.stop()

if __name__ == "__main__":
    main()


## https://wandb.ai/alexliusea2019-purdue-university/wandb_qilin/runs/79bwunr0?nw=nwuseralexliusea2019