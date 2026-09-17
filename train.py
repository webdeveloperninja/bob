"""A readable training loop: predict next byte, measure loss, update weights."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from model import TinyLM, choose_device


def batch(data, config, device, generator=None):
    length = config['context_length']
    starts = torch.randint(len(data) - length, (config['batch_size'],), generator=generator)
    chunks = np.stack([data[int(i):int(i) + length + 1] for i in starts])
    tokens = torch.from_numpy(chunks.astype(np.int64)).to(device)
    return tokens[:, :-1], tokens[:, 1:]


def main(args):
    if args.minutes is not None and (not math.isfinite(args.minutes) or args.minutes <= 0):
        raise ValueError('--minutes must be a finite positive number')
    torch.set_num_threads(4)
    device = choose_device(args.device)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    config = checkpoint['config'] if checkpoint else json.loads(Path(args.config).read_text())
    if args.steps is not None:
        config['steps'] = args.steps
    for key in ('context_length', 'width', 'layers', 'heads', 'batch_size', 'steps', 'eval_every', 'eval_batches'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if config['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    torch.manual_seed(config['seed'])
    data_path = Path(args.data)
    metadata = json.loads((data_path / 'meta.json').read_text())
    if checkpoint and checkpoint['data'] != metadata:
        raise ValueError('Resume requires the same prepared dataset as the checkpoint')
    data = {}
    for split in ('train', 'val'):
        path = data_path / f'{split}.bin'
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['splits'][split]['sha256']:
            raise ValueError(f'{split}.bin does not match meta.json')
        data[split] = np.memmap(path, dtype=np.uint8, mode='r')
        if len(data[split]) <= config['context_length']:
            raise ValueError(f'{split} is too short; prepare more data or reduce context_length')
    out = Path(args.out)
    if not checkpoint and out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --out or use --resume')
    out.mkdir(parents=True, exist_ok=True)
    model = TinyLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    start = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        torch.set_rng_state(checkpoint['rng'])
    if args.minutes is None and config['steps'] <= start:
        raise ValueError('--steps must be greater than the checkpoint step (it is a total)')
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    def precision():
        return torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; BF16: {use_bf16}', flush=True)
    started = time.monotonic()
    deadline = started + args.minutes * 60 if args.minutes is not None else None
    if deadline is not None:
        print(f'Training for approximately {args.minutes:g} additional minutes; starting at step {start}.', flush=True)
    step = start
    while True:
        step += 1
        model.train()
        x, y = batch(data['train'], config, device)
        optimizer.zero_grad(set_to_none=True)
        with precision():
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; lower learning_rate')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        finished = time.monotonic() >= deadline if deadline is not None else step >= config['steps']
        if step == 1 or step % config['eval_every'] == 0 or finished:
            model.eval()
            # Fixed validation windows make measurements comparable across steps.
            generator = torch.Generator().manual_seed(1234)
            with torch.no_grad(), precision():
                losses = []
                for _ in range(config['eval_batches']):
                    vx, vy = batch(data['val'], config, device, generator)
                    losses.append(F.cross_entropy(model(vx).reshape(-1, 256), vy.reshape(-1)).item())
            record = dict(step=step, train_loss=loss.item(), val_loss=sum(losses) / len(losses),
                          seconds=round(time.monotonic() - started, 2))
            print(json.dumps(record), flush=True)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            # Atomic replacement protects the previous checkpoint during saving.
            temporary = out / 'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            config=config, step=step, data=metadata, rng=torch.get_rng_state()), temporary)
            temporary.replace(out / 'checkpoint.pt')
            # Evaluation/saving count toward the budget too. The current step is saved.
            if finished or (deadline is not None and time.monotonic() >= deadline):
                break
    print(f'Saved {out / "checkpoint.pt"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/local.json')
    p.add_argument('--data', default='data/nemotron')
    p.add_argument('--out', default='runs/local')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    duration = p.add_mutually_exclusive_group()
    duration.add_argument('--steps', type=int, help='Total target steps, including previously completed steps')
    duration.add_argument('--minutes', type=float,
                          help='Train for this many additional minutes, ignoring config steps')
    p.add_argument('--resume', help='Path to checkpoint.pt; uses saved configuration')
    main(p.parse_args())
