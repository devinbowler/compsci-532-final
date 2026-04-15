import os
import time
import argparse
import csv
import random

from pipeline import get_spark, build_pipeline
from pyspark.sql import SparkSession
from queries import (
    total_word_count_df, total_word_count_sql,
    top_k_word_frequency_df, top_k_word_frequency_sql,
    top_k_word_pairs_df, top_k_word_pairs_sql,
    line_length_stats_df, line_length_stats_sql,
    high_frequency_filter_df, high_frequency_filter_sql,
)

def fresh_spark(partitions):
    return (
        SparkSession.builder
        .appName("Benchmark")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

def time_it(fn):
    t0 = time.time()
    result = fn()
    return time.time() - t0, result


def run_benchmark(spark, data_dir, top_k, threshold, use_cache, trial):
    spark.catalog.clearCache()
    lines_df, words_df, line_stats_df = build_pipeline(
        spark, data_dir, use_cache=use_cache
    )

    results = []

    def record(query, method, duration, trial):
        results.append({
            "query": query,
            "method": method,
            "time": duration,
            "cache": use_cache,
            "partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "trial": trial
        })

    # Q1
    t, _ = time_it(lambda: total_word_count_df(words_df))
    record("q1", "dataframe", t, trial)

    t, _ = time_it(lambda: total_word_count_sql(spark))
    record("q1", "sql", t, trial)

    # Q2
    t, _ = time_it(lambda: top_k_word_frequency_df(words_df, top_k).collect())
    record("q2", "dataframe", t, trial)

    t, _ = time_it(lambda: top_k_word_frequency_sql(spark, top_k).collect())
    record("q2", "sql", t, trial)

    # Q3
    t, _ = time_it(lambda: top_k_word_pairs_df(lines_df, top_k).collect())
    record("q3", "dataframe", t, trial)

    t, _ = time_it(lambda: top_k_word_pairs_sql(spark, lines_df, top_k).collect())
    record("q3", "sql", t, trial)

    # Q4
    t, _ = time_it(lambda: line_length_stats_df(line_stats_df).collect())
    record("q4", "dataframe", t, trial)

    t, _ = time_it(lambda: line_length_stats_sql(spark).collect())
    record("q4", "sql", t, trial)

    # Q5
    t, _ = time_it(lambda: high_frequency_filter_df(
        spark, lines_df, words_df, threshold
    ).count())
    record("q5", "dataframe", t, trial)

    t, _ = time_it(lambda: high_frequency_filter_sql(
        spark, threshold
    ).count())
    record("q5", "sql", t, trial)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results = []
    partitions_list = [1, 2, 3]

    for partitions in partitions_list:
        for cache in random.sample([True, False], 2):
            for trial in range(1, args.trials + 1):
                spark = fresh_spark(partitions)
                spark.sparkContext.setLogLevel("ERROR")

                print(f"partitions={partitions} cache={cache} trial={trial}")
                results = run_benchmark(
                    spark, args.data_dir,
                    top_k=args.top_k,
                    threshold=args.threshold,
                    use_cache=cache,
                    trial=trial,
                )
                all_results.extend(results)

        spark.stop()

    fieldnames = ["query", "method", "time", "cache", "partitions", "trial"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()