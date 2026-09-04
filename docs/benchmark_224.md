# RTX 2060 Mobile training benchmark

Measured locally on 2026-09-04 with PyTorch 2.7.1+cu126, ResNet-50,
10,000 outputs, batch size 32, 224x224 inputs, FP16 AMP, and full backward
and AdamW updates. Benchmarks use random initial weights, not production
checkpoints. They never save model weights.

Each run excludes model construction, file discovery, worker startup and
15 warm-up batches. CUDA is synchronized at the timing boundaries. Real
data runs use up to 8,192 images from 512 deterministically selected class
directories on G, with the existing CPU training augmentations. These are
short tests, not full-epoch or accuracy measurements.

| Input | Layout | AdamW | Workers | Timed steps | it/s | Images/s |
| --- | --- | --- | --- | --- | --- | --- |
| GPU-resident synthetic | NCHW | Default | N/A | 40 | 5.03 | 160.86 |
| GPU-resident synthetic | Channels last | Fused | N/A | 60 | 0.55 | 17.73 |
| GPU-resident synthetic | NCHW | Fused | N/A | 30 | 5.55 | 177.60 |
| Real images | NCHW | Fused | 4 | 60 | 5.05 | 161.59 |
| Real images | NCHW | Default | 4 | 60 | 5.12 | 163.76 |
| Real images | NCHW | Default | 6 | 100 | 4.81 | 153.94 |

GPU clock/power varied during testing; 855 MHz and about 45 W were observed
during the channels-last run. Do not attribute every difference exclusively
to layout or optimizer, or interpret the best short run as a sustained bound.

No reliable real-data improvement from fused AdamW or six workers was found.
Keep NCHW, default AdamW, four workers and CPU augmentation. Configure 224px
as requested. 15 it/s (480 images/s) was not achieved. Previous 192px results
are not directly comparable to 224px results.

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe benchmark_training.py --real --steps 100
.\.venv\Scripts\python.exe benchmark_training.py --real --fused --steps 100
```

Raw results are appended to `results/inaturalist/benchmarks/224_batch32.jsonl`.
Stop production training before benchmarking. No production training process
is started by the benchmark.
