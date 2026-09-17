"""Offline coverage of bounded pandas reading and GitHub text preparation."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from prepare import write_dataset
from prepare_code import frames, code_rows, raw_url


class CodeDataTests(unittest.TestCase):
    def test_pandas_filter_limits_and_training_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'metadata.parquet'
            pd.DataFrame([dict(repo='example/project', rel_path=f'file {i}.py',
                               language='Python' if i % 2 == 0 else 'CSS', commit_id='abcdef0')
                          for i in range(60)]).to_parquet(path)
            parquet = pq.ParquetFile(path)
            self.assertEqual(sum(len(df) for df in frames(parquet, 7)), 7)
            args = Namespace(out=str(root / 'data'), documents=20, max_rows=60,
                             max_fetches=30, language='python', max_bytes_per_document=1000, seed=42)
            sources, stats = [], dict(rows_scanned=0, fetch_attempts=0, skipped=0)
            def fake_fetch(session, url, limit):
                if 'file%200.py' in url:
                    raise ValueError('Unavailable file')
                return f'# {url}\ndef hello():\n    return 42\n'
            with patch('prepare_code.fetch_code', side_effect=fake_fetch):
                write_dataset(code_rows(parquet, args, sources, stats), args,
                              dict(dataset='fixture', fetched_sources=sources, retrieval=stats))
            meta = json.loads((root / 'data/meta.json').read_text())
            self.assertEqual(meta['splits']['train']['documents'], 18)
            self.assertEqual(meta['splits']['val']['documents'], 2)
            self.assertEqual(meta['retrieval']['skipped'], 1)
            self.assertEqual(len(meta['fetched_sources']), 20)
            self.assertIn(b'def hello():', (root / 'data/train.bin').read_bytes())
            self.assertTrue(all(row['language'] == 'Python' for row in sources))
            # Bound attempts even when there are not enough successes.
            args.max_fetches = 2
            stats = dict(rows_scanned=0, fetch_attempts=0, skipped=0)
            with patch('prepare_code.fetch_code', side_effect=fake_fetch):
                list(code_rows(parquet, args, [], stats))
            self.assertEqual(stats['fetch_attempts'], 2)

    def test_metadata_schema_and_urls(self):
        self.assertEqual(raw_url(dict(repo='owner/repo', rel_path='a b/file.py', commit_id='abcdef0')),
                         'https://raw.githubusercontent.com/owner/repo/abcdef0/a%20b/file.py')
        with self.assertRaises(ValueError):
            raw_url(dict(repo='owner/repo', rel_path='../file.py', commit_id='abcdef0'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.parquet'
            pd.DataFrame({'text': ['not code metadata']}).to_parquet(path)
            with self.assertRaises(ValueError):
                list(frames(pq.ParquetFile(path), 10))
