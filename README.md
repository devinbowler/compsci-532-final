# Spark SQL Text Analytics
**CS 532 – Systems for Data Science | Final Project**  
Devin Bowler · Henry Booth · Giordano Arcieri · Roberto Rubio

---

## Overview

This project builds a text analytics pipeline in PySpark that processes a corpus of ~4.5 million words across 5 large text files. We run 5 analytics queries on the data, each implemented two ways — using the **DataFrame API** and **Spark SQL** — to compare how both approaches perform on the same workload.

---

## Requirements

- Python 3.8+
- Java 11 (Java 17+ has compatibility issues with PySpark on some setups)
- PySpark 3.5.1

```bash
pip install pyspark==3.5.1
```

If you're on Windows, run everything through WSL to avoid Hadoop/winutils issues.

---

## How to run

```bash
python3 main.py
```

This loads all `.txt` files from the `data/` directory, runs all 5 queries both ways, and prints results to the terminal. A short summary is also saved to `results/query_results.txt` (the folder is created automatically if it doesn't exist).

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `./data` | directory containing corpus .txt files |
| `--top-k N` | `20` | how many results to show for word/pair rankings |
| `--threshold N` | `5000` | frequency cutoff for the high-frequency filter query |
| `--cache` | off | cache intermediate DataFrames before querying |
| `--partitions N` | `8` | shuffle partition count |

Example:
```bash
python3 main.py --top-k 10 --cache --partitions 16
```

---

## Project structure

```
project/
├── data/          # corpus .txt files (~24MB, ~4.5M words)
├── pipeline.py    # loads, cleans, tokenizes text and registers SQL views
├── queries.py     # all 5 queries, each in DataFrame and SQL versions
├── main.py        # runs everything, times queries, saves results summary
└── README.md
```

---

## Queries

| # | Query | What it does |
|---|-------|-------------|
| Q1 | Total word count | counts every word token in the corpus |
| Q2 | Top-k word frequency | most frequent individual words |
| Q3 | Top-k word pairs | most frequent adjacent word pairs per line |
| Q4 | Line length stats | avg / min / max words per line grouped by source file |
| Q5 | High-frequency filter | returns lines containing at least one word above the frequency threshold |

Each query has a `_df` version (DataFrame API) and a `_sql` version (Spark SQL) in `queries.py`. Both produce identical results — the point is to compare how they perform.

---

## Dataset

Five synthetic literary-style text files in `data/`, each modeled after a classic book title (War and Peace, Moby Dick, Les Miserables, Don Quixote, Shakespeare). Total corpus is ~4.55 million words across ~24MB of text. To use real Project Gutenberg books instead, drop plain `.txt` files into `data/` — no code changes needed.
