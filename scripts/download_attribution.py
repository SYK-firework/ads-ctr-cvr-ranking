"""Stream a subset of the Criteo Attribution Modeling for Bidding dataset from Hugging Face.

The full file is ~623MB compressed (~16.5M impressions over 30 days). Streaming reads rows
over HTTP and stops after --rows, so the full file is never stored on disk or in memory.
Output: a single Parquet file written batch by batch.

usage: python scripts/download_attribution.py --rows 2000000 --out data/attribution_2m.parquet
"""
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2_000_000)
    p.add_argument("--out", default="data/attribution_2m.parquet")
    p.add_argument("--batch", type=int, default=100_000)
    a = p.parse_args()

    ds = load_dataset("criteo/criteo-attribution-dataset", split="train", streaming=True)
    writer, n = None, 0
    for batch in ds.iter(batch_size=a.batch):
        table = pa.Table.from_pydict(batch)
        if n + len(table) > a.rows:
            table = table.slice(0, a.rows - n)
        if writer is None:
            writer = pq.ParquetWriter(a.out, table.schema)
            print("columns:", table.schema.names)
        writer.write_table(table)
        n += len(table)
        print(f"\r{n:,} / {a.rows:,} rows", end="", flush=True)
        if n >= a.rows:
            break
    if writer is not None:
        writer.close()
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
