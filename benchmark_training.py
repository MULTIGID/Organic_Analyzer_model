"""Bounded training benchmarks; never load or save model checkpoints."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.folder_multiclass import FileListDataset
from src.model import create_resnet50
from src.transforms import build_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', choices=['nchw', 'nhwc'], default='nchw')
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    fmt = torch.channels_last if args.layout == 'nhwc' else torch.contiguous_format
    model = create_resnet50(False, 10000).to('cuda', memory_format=fmt).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4,
                                 **({'fused': True} if args.fused else {}))
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    loader = None
    if args.real:
        root = Path('G:/Datasets/iNaturalist2021_full/dataset/train')
        folders = sorted(p for p in root.iterdir() if p.is_dir())
        rng = random.Random(42)
        samples = []
        for index in rng.sample(range(len(folders)), 512):
            files = list(folders[index].glob('*.jpg'))
            samples.extend((p, index) for p in rng.sample(files, min(16, len(files))))
        transform, _ = build_transforms(224)
        options = dict(batch_size=32, num_workers=args.workers, pin_memory=True,
                       shuffle=True, drop_last=True)
        if args.workers:
            options.update(persistent_workers=True, prefetch_factor=4)
        loader = DataLoader(FileListDataset(samples, transform), **options)
        iterator = iter(loader)
    else:
        images = torch.rand(32, 3, 224, 224, device='cuda').contiguous(memory_format=fmt)
        labels = torch.randint(10000, (32,), device='cuda')
    print('START', vars(args), 'threads', torch.get_num_threads(), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = None
    for step in range(args.steps + 15):
        if step == 15:
            torch.cuda.synchronize()
            started = time.perf_counter()
        if loader is not None:
            images, labels = next(iterator)
            images = images.to('cuda', non_blocking=True, memory_format=fmt)
            labels = labels.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = dict(**vars(args), threads_used=torch.get_num_threads(),
                  iterations_per_second=args.steps / seconds,
                  images_per_second=32 * args.steps / seconds,
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 1024**2,
                  final_loss=loss.item(), torch_version=torch.__version__)
    print(json.dumps(result), flush=True)
    target = Path('results/inaturalist/benchmarks')
    target.mkdir(parents=True, exist_ok=True)
    with (target / '224_batch32.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
