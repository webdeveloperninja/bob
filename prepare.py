"""Stream a bounded text sample and split documents BEFORE packing bytes."""
import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = 'nvidia/Nemotron-Pretraining-Dataset-sample'


def prepare(args):
    if args.documents < 20 or args.max_bytes_per_document < 1:
        raise ValueError('Use at least 20 documents and a positive byte limit')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; use a new --out directory')
    if args.text_file:
        # One nonempty line = one document, useful for offline experiments.
        rows = ({'text': line} for line in Path(args.text_file).read_text(encoding='utf-8').splitlines())
        revision = None
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
        rows = load_dataset(DATASET, args.subset, split='train',
                            streaming=True, revision=revision)
    metadata = dict(dataset=DATASET if not args.text_file else str(args.text_file),
                    subset=args.subset if not args.text_file else None, revision=revision)
    write_dataset(rows, args, metadata)


def write_dataset(rows, args, metadata):
    """Shared byte preparation for text documents and fetched code files."""
    out = Path(args.out)
    if args.documents < 20 or args.max_bytes_per_document < 1:
        raise ValueError('Use at least 20 documents and a positive byte limit')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; use a new --out directory')
    documents, seen = [], set()
    for row in rows:
        text = row.get('text')
        if not isinstance(text, str):
            raise ValueError('This subset does not have a string text field; use a text subset')
        blob = text.strip().encode('utf-8')[:args.max_bytes_per_document]
        digest = hashlib.sha256(blob).hexdigest()
        if not blob or digest in seen:
            continue
        seen.add(digest)
        documents.append(blob + b'\n\n')
        if len(documents) >= args.documents:
            break
    if len(documents) < 20:
        raise ValueError(f'Only {len(documents)} unique documents; need at least 20')
    random.Random(args.seed).shuffle(documents)
    n_val = max(1, len(documents) // 10)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, seed=args.seed, tokenizer='utf8-bytes-v1',
                    max_bytes_per_document=args.max_bytes_per_document, splits={})
    for name, docs in [('val', documents[:n_val]), ('train', documents[n_val:])]:
        blob = b''.join(docs)
        (out / f'{name}.bin').write_bytes(blob)
        metadata['splits'][name] = dict(documents=len(docs), tokens=len(blob),
                                        sha256=hashlib.sha256(blob).hexdigest())
    (out / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset', default='Nemotron-CC-High-Quality')
    p.add_argument('--revision', default='main')
    p.add_argument('--documents', type=int, default=1000)
    p.add_argument('--max-bytes-per-document', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='data/nemotron')
    p.add_argument('--text-file', help='Offline alternative: one document per line')
    prepare(p.parse_args())
