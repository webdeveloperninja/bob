"""Inspect NVIDIA code metadata with pandas, or fetch a small code training set."""
import argparse
from contextlib import contextmanager
import hashlib
import re
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, HfFileSystem

from prepare import write_dataset

REPO = 'nvidia/Nemotron-Pretraining-Code-v3'
FILE = 'Nemotron-Code-Metadata/part_00000.parquet'
COLUMNS = ['repo', 'rel_path', 'language', 'commit_id']


@contextmanager
def metadata_file(revision, filename):
    # Range reads avoid loading the entire shard into a DataFrame on a laptop.
    fs = HfFileSystem()
    with fs.open(f'datasets/{REPO}@{revision}/{filename}', 'rb') as f:
        yield pq.ParquetFile(f)


def frames(parquet, max_rows):
    missing = set(COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'Metadata is missing columns: {sorted(missing)}')
    scanned = 0
    for batch in parquet.iter_batches(batch_size=min(1000, max_rows), columns=COLUMNS):
        df = batch.to_pandas().head(max_rows - scanned)
        yield df
        scanned += len(df)
        if scanned >= max_rows:
            break


def raw_url(row):
    repo, path, commit = row['repo'], row['rel_path'], row['commit_id']
    if not all(isinstance(value, str) for value in (repo, path, commit)):
        raise ValueError('Invalid metadata strings')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Invalid repository name')
    if not re.fullmatch(r'[0-9a-fA-F]{7,40}', commit):
        raise ValueError('Invalid commit ID')
    if not path or path.startswith('/') or any(part in ('.', '..') for part in path.split('/')):
        raise ValueError('Invalid repository-relative path')
    return f'https://raw.githubusercontent.com/{repo}/{commit}/{quote(path, safe="/")}'


def fetch_code(session, url, limit):
    with session.get(url, stream=True, timeout=(10, 20)) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content.extend(chunk[:limit - len(content)])
            if len(content) >= limit:
                break
    if b'\x00' in content:
        raise ValueError('Binary file')
    # A byte limit can cut a UTF-8 character. Match prepare.py's byte-based cap.
    return bytes(content).decode('utf-8', errors='replace')


def code_rows(parquet, args, provenance, stats):
    with requests.Session() as session:
        for df in frames(parquet, args.max_rows):
            stats['rows_scanned'] += len(df)
            if args.language:
                df = df[df['language'].str.casefold() == args.language.casefold()]
            for row in df.to_dict(orient='records'):
                if stats['fetch_attempts'] >= args.max_fetches:
                    return
                stats['fetch_attempts'] += 1
                try:
                    url = raw_url(row)
                    text = fetch_code(session, url, args.max_bytes_per_document)
                except (requests.RequestException, ValueError) as exc:
                    stats['skipped'] += 1
                    print(f'Skipped candidate {stats["fetch_attempts"]}: {type(exc).__name__}', flush=True)
                    continue
                if not text.strip():
                    stats['skipped'] += 1
                    continue
                provenance.append(dict(row, url=url, sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
                yield {'text': text}


def main(args):
    if min(args.max_rows, args.max_fetches, args.max_bytes_per_document) < 1 or args.documents < 20:
        raise ValueError('Limits must be positive and --documents must be at least 20')
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    print(f'Dataset revision: {revision}', flush=True)
    with metadata_file(revision, args.file) as parquet:
        if args.inspect:
            for df in frames(parquet, min(args.max_rows, 10)):
                with pd.option_context('display.max_colwidth', 60, 'display.width', 160):
                    print(df.to_string(index=False))
            print('Metadata only: use this script without --inspect to fetch code from GitHub.')
            return
        provenance = []
        stats = dict(rows_scanned=0, fetch_attempts=0, skipped=0)
        metadata = dict(dataset=REPO, revision=revision, file=args.file,
                        language=args.language, max_rows=args.max_rows, max_fetches=args.max_fetches,
                        fetched_sources=provenance, retrieval=stats)
        rows = code_rows(parquet, args, provenance, stats)
        try:
            write_dataset(rows, args, metadata)
        except ValueError as exc:
            raise ValueError(f'{exc}. Scanned {stats["rows_scanned"]} metadata rows, '
                             f'attempted {stats["fetch_attempts"]} fetches. '
                             'Try another shard/language or increase the limits.') from exc
        finally:
            rows.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect', action='store_true', help='Show up to 10 metadata rows; no GitHub downloads or output files')
    p.add_argument('--file', default=FILE, help='Parquet shard path within the NVIDIA dataset')
    p.add_argument('--revision', default='main', help='Dataset revision, resolved and recorded as a commit')
    p.add_argument('--language', help='Case-insensitive language filter; default keeps all languages')
    p.add_argument('--documents', type=int, default=100, help='Maximum unique code documents to keep (at least 20)')
    p.add_argument('--max-rows', type=int, default=10000, help='Maximum metadata rows to examine before filtering')
    p.add_argument('--max-fetches', type=int, default=300, help='Maximum GitHub file requests')
    p.add_argument('--max-bytes-per-document', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='data/nemotron-code')
    args = p.parse_args()
    try:
        main(args)
    except ValueError as exc:
        p.error(str(exc))
