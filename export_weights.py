"""Export every model weight as readable JSON text (no truncated tensors)."""
import argparse
import json
from pathlib import Path

import torch


def export(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint}')
    if output.exists():
        raise ValueError(f'Output already exists: {output}; choose another --out path')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = saved['model']
    count = sum(t.numel() for t in weights.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        # .tolist() includes EVERY value, unlike PyTorch's abbreviated tensor printout.
        json.dump({
            'checkpoint': str(checkpoint),
            'step': saved['step'],
            'parameter_count': count,
            'config': saved['config'],
            'weights': {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                               'values': t.tolist()} for name, t in weights.items()},
        }, f, indent=2, allow_nan=False)
        f.write('\n')
    print(f'Exported {count:,} parameters to {output} ({output.stat().st_size / 1_000_000:.1f} MB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/local/checkpoint.pt')
    p.add_argument('--out', default='runs/local/weights.txt')
    args = p.parse_args()
    try:
        export(args.checkpoint, args.out)
    except ValueError as exc:
        p.error(str(exc))
