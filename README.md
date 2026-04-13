# Text analytics with PySpark

This is a small final project that reads a bunch of plain `.txt` books, runs them through Apache Spark (PySpark), and prints some stats: total words, most common words, common word pairs, per-file line stats, and a filter for lines that contain really common words.

Nothing fancy—mostly practice using Spark both the “DataFrame API” way and the “write SQL strings” way for the same ideas.

## What you need

- Python 3 with **PySpark** installed (`pip install pyspark` or whatever you already use).
- A **Java version Spark actually likes**. On Windows I had issues with super new Java (like 25); **Java 17 or 21** is the safe bet. On Linux/WSL the stock OpenJDK is usually fine.

Put your books as `.txt` files in the `data/` folder (or point the script somewhere else—see below).

## How to run it

From this folder:

```bash
python main.py
```

On Linux you might use `python3`:

```bash
python3 main.py
```

That uses the default `data/` directory next to `main.py`, runs five “queries,” prints a bunch of tables to the terminal, and writes a short text summary to `../results/query_results.txt` (one folder above this project, in a `results` folder—it creates the folder if it’s missing).

### Optional flags (if you want to mess with stuff)

| Flag | What it does |
|------|----------------|
| `--data-dir PATH` | Where to look for `.txt` files (default: `./data`) |
| `--top-k N` | How many rows to show for word / pair rankings (default: 20) |
| `--threshold N` | Cutoff for the “high frequency word” filter (default: 5000) |
| `--cache` | Tells Spark to cache some DataFrames (can help if you re-run steps; first run might still feel slow) |
| `--partitions N` | Shuffle partitions (default: 8); tweak if you’re bored |

Example:

```bash
python main.py --top-k 10 --threshold 3000
```

## What the code files are

- **`pipeline.py`** — Starts Spark, reads the text files, cleans lines, splits into words, registers temp views for SQL.
- **`queries.py`** — The actual analytics: each query has a DataFrame version and a Spark SQL version so you can compare.
- **`main.py`** — Glue code: parses args, runs everything in order, times stuff, saves the little results file.

Spark will spam some warnings (especially on Windows about Hadoop paths). If the job still runs, you can ignore most of that.

## Heads up

The book files in `data/` are big; cloning or pushing can take a bit. If your machine is slow, start with one small `.txt` in `data/` and `--data-dir` pointing at it.

That’s basically it—run `main.py`, read the console output, check `../results/query_results.txt` if you want the saved summary.
