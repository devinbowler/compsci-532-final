import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode, split, lower, regexp_replace, trim, length, size, input_file_name, regexp_extract


# create a spark session with configurable shuffle partitions
def get_spark(app_name="TextAnalytics", partitions=8):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .getOrCreate()
    )


# read all text files and tag each line with its source filename
def load_texts(spark, paths):
    if isinstance(paths, str):
        paths = [paths]

    raw = spark.read.text(paths)

    df = raw.withColumnRenamed("value", "raw_line").withColumn(
        "source",
        regexp_extract(input_file_name(), r"([^/\\]+)$", 1),
    ).select("source", "raw_line")

    return df


# lowercase, strip punctuation, collapse whitespace, drop empty lines
def clean_lines(df):
    cleaned = (
        df
        .withColumn("clean_line", lower(col("raw_line")))
        .withColumn("clean_line", regexp_replace(col("clean_line"), r"[^a-z0-9\s]", ""))
        .withColumn("clean_line", regexp_replace(col("clean_line"), r"\s+", " "))
        .withColumn("clean_line", trim(col("clean_line")))
        .filter(col("clean_line") != "")
        .select("source", "clean_line")
    )
    return cleaned


# split each line into individual word rows
def tokenize(df):
    words = (
        df
        .select("source", explode(split(col("clean_line"), r"\s+")).alias("word"))
        .filter(col("word") != "")
    )
    return words


# add word count and character count columns to each line
def add_line_stats(df):
    return (
        df
        .withColumn("word_count", size(split(col("clean_line"), r"\s+")))
        .withColumn("char_count", length(col("clean_line")))
    )


# register dataframes as sql temp views so we can query them with spark sql
def register_views(lines_df, words_df, line_stats_df):
    lines_df.createOrReplaceTempView("lines")
    words_df.createOrReplaceTempView("words")
    line_stats_df.createOrReplaceTempView("line_stats")


# run the full pipeline: load all txt files, clean, tokenize, register views
def build_pipeline(spark, data_dir, use_cache=False):
    if os.path.isfile(data_dir):
        txt_files = [data_dir]
    else:
        txt_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".txt") and os.path.getsize(os.path.join(data_dir, f)) > 0
        ]

    if not txt_files:
        raise FileNotFoundError(f"no .txt files found in {data_dir}")

    print(f"loading {len(txt_files)} files from {data_dir}")

    raw_df        = load_texts(spark, txt_files)
    lines_df      = clean_lines(raw_df)
    words_df      = tokenize(lines_df)
    line_stats_df = add_line_stats(lines_df)

    if use_cache:
        lines_df.cache()
        words_df.cache()
        line_stats_df.cache()

    register_views(lines_df, words_df, line_stats_df)

    return lines_df, words_df, line_stats_df