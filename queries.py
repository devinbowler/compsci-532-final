from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, desc, avg, min as spark_min, max as spark_max, explode, split, udf
from pyspark.sql.types import ArrayType, StringType


# udf wrapped in a factory so spark workers can serialize it without needing to import this module
def _make_pairs_udf():
    def get_pairs(words):
        if not words or len(words) < 2:
            return []
        pairs = []
        for i in range(len(words) - 1):
            a, b = sorted([words[i], words[i + 1]])
            if a and b:
                pairs.append(f"{a}|{b}")
        return pairs
    return udf(get_pairs, ArrayType(StringType()))

pairs_udf = _make_pairs_udf()


# query 1: total word count

def total_word_count_df(words_df):
    return words_df.count()

def total_word_count_sql(spark):
    result = spark.sql("SELECT COUNT(*) AS total FROM words")
    return result.collect()[0]["total"]


# query 2: top k most frequent words

def top_k_word_frequency_df(words_df, k=20):
    return (
        words_df
        .groupBy("word")
        .count()
        .orderBy(desc("count"))
        .limit(k)
    )

def top_k_word_frequency_sql(spark, k=20):
    return spark.sql(f"""
        SELECT word, COUNT(*) AS count
        FROM words
        GROUP BY word
        ORDER BY count DESC
        LIMIT {k}
    """)


# query 3: top k most frequent adjacent word pairs per line

def top_k_word_pairs_df(lines_df, k=20):
    return (
        lines_df
        .select(explode(pairs_udf(split(col("clean_line"), r"\s+"))).alias("pair"))
        .filter(col("pair") != "")
        .groupBy("pair")
        .count()
        .orderBy(desc("count"))
        .limit(k)
    )

def top_k_word_pairs_sql(spark, lines_df, k=20):
    # build the pairs view first since sql can't call the udf directly
    pairs_df = lines_df.select(
        explode(pairs_udf(split(col("clean_line"), r"\s+"))).alias("pair")
    ).filter(col("pair") != "")

    pairs_df.createOrReplaceTempView("pairs")

    return spark.sql(f"""
        SELECT pair, COUNT(*) AS count
        FROM pairs
        GROUP BY pair
        ORDER BY count DESC
        LIMIT {k}
    """)


# query 4: avg, min, max words per line grouped by source file

def line_length_stats_df(line_stats_df):
    return (
        line_stats_df
        .groupBy("source")
        .agg(
            avg("word_count").alias("avg_words"),
            spark_min("word_count").alias("min_words"),
            spark_max("word_count").alias("max_words")
        )
        .orderBy("source")
    )

def line_length_stats_sql(spark):
    return spark.sql("""
        SELECT source,
               AVG(word_count) AS avg_words,
               MIN(word_count) AS min_words,
               MAX(word_count) AS max_words
        FROM line_stats
        GROUP BY source
        ORDER BY source
    """)


# query 5: return lines that contain at least one word above the frequency threshold

def high_frequency_filter_df(spark, lines_df, words_df, threshold=5000):
    # find all words that appear more than threshold times across the corpus
    hot_words = (
        words_df
        .groupBy("word")
        .count()
        .filter(col("count") > threshold)
        .select("word")
    )

    # explode lines into words, join against hot words, then get back the original lines
    words_with_lines = lines_df.select(
        "source", "clean_line",
        explode(split(col("clean_line"), r"\s+")).alias("word")
    )

    return (
        words_with_lines
        .join(hot_words, on="word", how="inner")
        .select("source", "clean_line")
        .distinct()
        .orderBy("source", "clean_line")
    )

def high_frequency_filter_sql(spark, threshold=5000):
    return spark.sql(f"""
        SELECT DISTINCT l.source, l.clean_line
        FROM lines l
        JOIN (
            SELECT word
            FROM words
            GROUP BY word
            HAVING COUNT(*) > {threshold}
        ) hot
          ON l.clean_line LIKE CONCAT('% ', hot.word, ' %')
          OR l.clean_line LIKE CONCAT(hot.word, ' %')
          OR l.clean_line LIKE CONCAT('% ', hot.word)
          OR l.clean_line = hot.word
        ORDER BY l.source, l.clean_line
    """)