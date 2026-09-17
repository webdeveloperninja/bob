"""Offline checks for causality, learning, and checkpoint continuation."""
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
import json

import torch
from torch.nn import functional as F
from model import TinyLM
from export_weights import export


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = dict(context_length=16, width=32, layers=1, heads=2,
                           batch_size=2, steps=2, learning_rate=0.003,
                           eval_every=1, eval_batches=1, seed=42)

    def test_future_tokens_cannot_change_past_predictions(self):
        model = TinyLM(self.config).eval()
        x = torch.randint(256, (2, 16))
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 256
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :8], model(changed)[:, :8])

    def test_can_learn_repeated_pattern(self):
        model = TinyLM(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.tensor([[65, 66] * 8])
        y = torch.tensor([[66, 65] * 8])
        initial = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.2)

    def test_prepare_train_resume_generate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'text.txt'
            source.write_text('\n'.join(f'Document {i}: learning the next byte. ' * 8 for i in range(30)))
            config = folder / 'config.json'
            config.write_text(json.dumps(self.config))
            data, run = folder / 'data', folder / 'run'
            def command(*args):
                return subprocess.run([sys.executable, *map(str, args)], cwd=root,
                                      check=True, capture_output=True, text=True).stdout
            command('prepare.py', '--text-file', source, '--out', data)
            command('train.py', '--config', config, '--data', data, '--out', run, '--device', 'cpu')
            checkpoint = run / 'checkpoint.pt'
            command('train.py', '--resume', checkpoint, '--data', data, '--out', run,
                    '--steps', '3', '--device', 'cpu')
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['step'], 3)
            # Same CPU run continued from checkpoint should match uninterrupted training.
            full = folder / 'full'
            command('train.py', '--config', config, '--data', data, '--out', full,
                    '--steps', '3', '--device', 'cpu')
            reference = torch.load(full / 'checkpoint.pt', weights_only=True)
            for key in saved['model']:
                torch.testing.assert_close(saved['model'][key], reference['model'][key])
            output = command('generate.py', '--checkpoint', checkpoint, '--prompt', 'Hello',
                             '--bytes', '10', '--device', 'cpu')
            self.assertTrue(output.startswith('Hello'))
            # The duration overrides the saved step limit, even on a completed run.
            timed = folder / 'timed'
            command('train.py', '--resume', checkpoint, '--data', data, '--out', timed,
                    '--minutes', '0.002', '--device', 'cpu')
            continued = torch.load(timed / 'checkpoint.pt', weights_only=True)
            self.assertGreater(continued['step'], saved['step'])
            records = [json.loads(line) for line in (timed / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['step'], continued['step'])
            text_file = folder / 'weights.txt'
            export(timed / 'checkpoint.pt', text_file)
            exported = json.loads(text_file.read_text())
            self.assertEqual(exported['parameter_count'], sum(t.numel() for t in continued['model'].values()))
            for name, tensor in continued['model'].items():
                self.assertEqual(exported['weights'][name]['shape'], list(tensor.shape))
                torch.testing.assert_close(torch.tensor(exported['weights'][name]['values']), tensor,
                                           rtol=0, atol=0)
            with self.assertRaises(ValueError):
                export(timed / 'checkpoint.pt', timed / 'checkpoint.pt')


if __name__ == '__main__':
    unittest.main()
