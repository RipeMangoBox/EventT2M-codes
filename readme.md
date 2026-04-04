# Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis (ICLR 2026)

The official PyTorch implementation of the paper "Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis".

## Setting an Environment

### 1. Create Conda Environment

```bash
conda create -n event-t2m python==3.10.14
conda activate event-t2m

# install pytorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


# install requirements
pip install -r requirements.txt
```

### 2. Download the Original Datasets

We conduct experiments on the HumanML3D and KIT-ML datasets. For both datasets, you can download them by following the instructions in [here](https://github.com/EricGuo5513/HumanML3D).

### 3. Prepare the HumanML3D-E Dataset

You can download the completed HumanML3D-E dataset from [here](https://drive.google.com/drive/folders/19mPyYV8j1vnfJ6W9tZX9758JtDUQpYop?usp=sharing).

If you want to prepare the dataset from scratch, follow the steps below:

Since an LLM (Gemini 2.5 flash) was used for HumanML3D-E data preprocessing, an API key is required.
Please enter the issued API key on line 6 of `src/tools/data_decompose.py`.

```bash
GOOGLE_API_KEY = "" # your api key here
```

- For processing,

```bash
python src/tools/data_decompose.py
```

### 4. Preprocess the Datasets

```bash
python src/tools/data_preprocess_decomposed.py --dataset hml3d
python src/tools/data_preprocess_decomposed.py --dataset kit
```

This will add the following files to the directory:
```
./dataset/HumanML3D
├── ...
├── data_train.npy
├── data_val.npy
└── data_test.npy
```

Also, we have released test subsets based on the number of conditions for event-stratified evaluation.

```
./dataset/HumanML3D
├── ...
├── data_test_condition2.npy
├── data_test_condition3.npy
└── data_test_condition4.npy
```

### 5. Download Dependencies and Pre-trained Models

Download and unzip dependencies from [here](https://onedrive.live.com/?id=76593CF7B7FC849C%21180700&resid=76593CF7B7FC849C%21180700&e=345HR5&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcHlFX0xmM1BGbDJpNE5jRThtZ1ZVTjNvWDluVFE_ZT0zNDVIUjU&cid=76593cf7b7fc849c&v=validatepermission).

Download and unzip pre-trained models from [here](https://drive.google.com/drive/folders/19mPyYV8j1vnfJ6W9tZX9758JtDUQpYop?usp=sharing).

```
./
├── checkpoints
||   ├── hml3d.ckpt
||   ├── kit.ckpt
├── deps
||   ├── glove
||   ├── t2m_guo
└── ...
```

## Training

- For HumanML3D

```bash
python src/train.py trainer.devices="[0]" logger=wandb data=hml3d_event_final \
    data.batch_size=512 data.repeat_dataset=5 trainer.max_epochs=600 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed
```

- For KIT-ML

```bash
python src/train.py trainer.devices="[2,3]" logger=wandb data=kit_event_final \
    data.batch_size=512 data.repeat_dataset=5 trainer.max_epochs=1000 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed
```

### Training hyperparameters

Commonly useful knobs (Hydra overrides):

- `model.text_encoder.cache_text_embeddings=true` (default: `true`)
  - Cache TMR text embeddings in memory across training steps.
  - Safe because `encode_text` uses `sample_mean=True` — the output is fully deterministic given the same text and weights, regardless of device, random seed, or run.
  - Eliminates redundant frozen text encoder forward passes and reduces per-step overhead noticeably at large batch sizes.
  - Set to `false` to disable caching, e.g. when debugging the text encoder or if you modify its behaviour to introduce stochasticity.
- `data.batch_size=512`
  - training batch size
- `data.repeat_dataset=5`
  - number of times the dataset is repeated per epoch (effectively multiplies steps per epoch)
- `trainer.max_epochs=600`
  - total training epochs
- `model.guidance_scale=4`
  - classifier-free guidance scale
- `model.noise_scheduler.prediction_type=sample`
  - diffusion prediction type
- `trainer.precision=bf16-mixed`
  - mixed precision mode; `bf16-mixed` is recommended for modern GPUs

## Evaluation

This workspace supports two evaluation modes through `src/eval.py` only. The helper shell wrapper has been removed on purpose, so please run evaluation directly with the commands below.

### Mode 1: TMR-aligned retrieval only

Use this mode when you only want fair retrieval comparison against TMR / MotionPatches. It skips EventT2M's native diffusion evaluation and exports only the TMR-aligned retrieval YAMLs.

For HumanML3D:

```bash
python src/eval.py trainer.devices="[0]" data=hml3d_event_final data.test_batch_size=128 \
    model=event_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample \
    model.denoiser.stage_dim="256*4" \
    ckpt_path="checkpoints/pretrained/HumanML3D/hml3d.ckpt" \
    retrieval_only=true model.metrics.enable_mm_metric=false
```

For KIT-ML:

```bash
python src/eval.py trainer.devices="[0]" data=kit_event_final data.test_batch_size=128 \
    model=event_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample \
    model.denoiser.stage_dim="256*4" \
    ckpt_path="checkpoints/pretrained/KIT-ML/kit.ckpt" \
    retrieval_only=true model.metrics.enable_mm_metric=false
```

This mode writes the following TMR-aligned retrieval files into the checkpoint folder's `eval/` directory:

- `normal.yaml`
- `threshold_0.95.yaml`
- `nsim.yaml`
- `guo.yaml`

No EventT2M native diffusion report is produced in this mode.

### Mode 2: TMR-aligned retrieval + EventT2M native evaluation

Use this mode when you want both:

- the same TMR-aligned retrieval export used for cross-repository comparison; and
- EventT2M's original native generation evaluation.

For HumanML3D:

```bash
python src/eval.py trainer.devices="[0]" data=hml3d_event_final data.test_batch_size=128 \
    model=event_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample \
    model.denoiser.stage_dim="256*4" \
    ckpt_path="checkpoints/pretrained/HumanML3D/hml3d.ckpt" \
    retrieval_only=false model.metrics.enable_mm_metric=false
```

For KIT-ML:

```bash
python src/eval.py trainer.devices="[0]" data=kit_event_final data.test_batch_size=128 \
    model=event_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample \
    model.denoiser.stage_dim="256*4" \
    ckpt_path="checkpoints/pretrained/KIT-ML/kit.ckpt" \
    retrieval_only=false model.metrics.enable_mm_metric=false
```

This mode writes:

- TMR-aligned retrieval YAMLs:
  - `normal.yaml`
  - `threshold_0.95.yaml`
  - `nsim.yaml`
  - `guo.yaml`
- EventT2M native diffusion evaluation:
  - `E-native_normal.yaml`
- raw native metric dump:
  - `metrics.json`

### Diffusion / native evaluation hyperparameters

When `retrieval_only=false`, the native EventT2M diffusion evaluation is active. You can control it from the command line with Hydra overrides.

Commonly useful knobs:

- `model.guidance_scale=4`
  - classifier-free guidance scale used during diffusion sampling
- `model.step_num=10`
  - number of diffusion sampling steps
- `model.noise_scheduler.prediction_type=sample`
  - diffusion prediction type
- `model.metrics.replicate_times=20`
  - number of repeated native evaluations used to report mean and confidence interval
- `model.metrics.enable_mm_metric=false`
  - whether to compute multimodality metrics; turning this on makes evaluation much slower
- `model.metrics.mm_num_samples=100`
  - number of samples used when multimodality evaluation is enabled
- `data.test_batch_size=128`
  - dataloader batch size for native evaluation
- `trainer.devices="[0]"`
  - GPU device selection

Example with explicit diffusion controls:

```bash
python src/eval.py trainer.devices="[0]" data=hml3d_event_final data.test_batch_size=64 \
    model=event_final \
    model.guidance_scale=5 model.step_num=20 \
    model.noise_scheduler.prediction_type=sample \
    model.metrics.replicate_times=5 model.metrics.enable_mm_metric=false \
    ckpt_path="checkpoints/pretrained/HumanML3D/hml3d.ckpt" \
    retrieval_only=false
```

### Notes

- `retrieval_only=true` means: export only TMR-aligned retrieval metrics.
- `retrieval_only=false` means: run EventT2M native diffusion evaluation first, then export TMR-aligned retrieval metrics.
- The retrieval YAML naming is intentionally aligned with TMR: `normal.yaml`, `threshold_0.95.yaml`, `nsim.yaml`, `guo.yaml`.
- The `E-` prefix in `E-native_normal.yaml` is used to distinguish EventT2M native output from TMR-style retrieval output.
- Setting `model.metrics.enable_mm_metric=true` will significantly increase runtime.
- If your environment uses a different GPU syntax, adjust `trainer.devices` accordingly.

