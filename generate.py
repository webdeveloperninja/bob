"""Continue a prompt using your trained weights."""
import argparse
from pathlib import Path
import sys
import torch
from model import TinyLM, choose_device


def main(args):
    if not Path(args.checkpoint).is_file():
        message = (
            f'Checkpoint not found: {args.checkpoint}\n'
            'Run python train.py to create runs/local/checkpoint.pt, or pass\n'
            '--checkpoint PATH to use an existing trained model.'
        )
        if Path('runs/first-check/checkpoint.pt').is_file():
            message += (
                '\n\nThe 20-step sample is available. Try:\n'
                'python generate.py --checkpoint runs/first-check/checkpoint.pt '
                '--prompt "The future of AI"'
            )
        sys.exit(message)
    if not args.prompt or args.temperature <= 0 or args.bytes < 0 or not 1 <= args.top_k <= 256:
        raise ValueError('Use a nonempty prompt, positive temperature, bytes >= 0, and top-k in 1..256')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TinyLM(saved['config']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    tokens = torch.tensor([list(args.prompt.encode('utf-8'))], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(args.bytes):
            logits = model(tokens[:, -model.context_length:])[:, -1, :] / args.temperature
            values, indices = torch.topk(logits, args.top_k)
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
            tokens = torch.cat([tokens, indices.gather(-1, choice)], dim=1)
    print(bytes(tokens[0].tolist()).decode('utf-8', errors='replace'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/local/checkpoint.pt')
    p.add_argument('--prompt', default='The future of artificial intelligence')
    p.add_argument('--bytes', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    main(p.parse_args())
