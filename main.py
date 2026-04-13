import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# import pipeline and all query functions
from pipeline import get_spark, build_pipeline
from queries import (
    total_word_count_df, total_word_count_sql,
    top_k_word_frequency_df, top_k_word_frequency_sql,
    top_k_word_pairs_df, top_k_word_pairs_sql,
    line_length_stats_df, line_length_stats_sql,
    high_frequency_filter_df, high_frequency_filter_sql,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--top-k",      type=int, default=20)
    p.add_argument("--threshold",  type=int, default=5000)
    p.add_argument("--cache",      action="store_true")
    p.add_argument("--partitions", type=int, default=8)
    return p.parse_args()


# print a section header
def section(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print('=' * 50)


# run a function and print how long it took
def timed(label, fn):
    t0 = time.time()
    result = fn()
    print(f"  {label} finished in {time.time() - t0:.3f}s")
    return result


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "query_results.txt")

    spark = get_spark(partitions=args.partitions)
    spark.sparkContext.setLogLevel("WARN")

    try:
        lines_df, words_df, line_stats_df = build_pipeline(spark, data_dir, use_cache=args.cache)

        output_lines = []

        # query 1: total word count
        section("query 1 - total word count")
        count_df  = timed("dataframe", lambda: total_word_count_df(words_df))
        count_sql = timed("sql", lambda: total_word_count_sql(spark))
        print(f"  dataframe: {count_df:,}")
        print(f"  sql:       {count_sql:,}")
        output_lines.append(f"q1 dataframe: {count_df}")
        output_lines.append(f"q1 sql: {count_sql}")

        # query 2: top k word frequency
        section(f"query 2 - top {args.top_k} word frequency")
        freq_df  = timed("dataframe", lambda: top_k_word_frequency_df(words_df, args.top_k))
        freq_sql = timed("sql", lambda: top_k_word_frequency_sql(spark, args.top_k))
        freq_df.show(truncate=False)
        freq_sql.show(truncate=False)

        # query 3: top k word pairs
        section(f"query 3 - top {args.top_k} word pairs")
        pairs_df  = timed("dataframe", lambda: top_k_word_pairs_df(lines_df, args.top_k))
        pairs_sql = timed("sql", lambda: top_k_word_pairs_sql(spark, lines_df, args.top_k))
        pairs_df.show(truncate=False)
        pairs_sql.show(truncate=False)

        # query 4: line length stats per source file
        section("query 4 - line length stats")
        stats_df  = timed("dataframe", lambda: line_length_stats_df(line_stats_df))
        stats_sql = timed("sql", lambda: line_length_stats_sql(spark))
        stats_df.show(truncate=False)
        stats_sql.show(truncate=False)

        # query 5: filter lines containing high frequency words
        section(f"query 5 - high frequency filter (threshold={args.threshold})")
        hff_df  = timed("dataframe", lambda: high_frequency_filter_df(spark, lines_df, words_df, args.threshold))
        hff_sql = timed("sql", lambda: high_frequency_filter_sql(spark, args.threshold))
        df_count  = hff_df.count()
        sql_count = hff_sql.count()
        print(f"  dataframe matching lines: {df_count:,}")
        print(f"  sql matching lines:       {sql_count:,}")
        hff_df.show(5, truncate=80)
        output_lines.append(f"q5 dataframe matching lines: {df_count}")
        output_lines.append(f"q5 sql matching lines: {sql_count}")

        # write summary to file
        with open(results_path, "w") as f:
            f.write("query results\n")
            f.write("=" * 30 + "\n")
            for line in output_lines:
                f.write(line + "\n")

        print(f"\nresults written to {results_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()