# run\_sweep_autoencoder.py Documentation

This document explains the purpose, installation, and usage of the `run_sweep_autoencoder.py` script, which automates Weights & Biases (W\&B) sweep launches from a YAML configuration file.

---

## Overview

`run_sweep_autoencoder.py` is a command-line utility that:

1. **Parses** a sweep configuration YAML path, W\&B project name, and number of runs from CLI arguments.
2. **Loads** the YAML to construct a W\&B sweep configuration.
3. **Injects** a `device` parameter (e.g. `cuda:0` or `cpu`) based on local hardware.
4. **Starts** the sweep via `wandb.sweep`.
5. **Launches** multiple runs of your `train_sweep` function with `wandb.agent`.

This allows you to quickly launch, track, and manage hyperparameter sweeps with minimal boilerplate.

---

## CLI Usage

Make the script executable:

```bash
tchmod +x run_sweep_autoencoder.py
```

Then run:

```bash
# Default config, project, and 10 runs
./run_sweep_autoencoder.py

# Custom config, project 'my-project', 5 runs
./run_sweep_autoencoder.py \
  --config /path/to/my_sweep.yaml \
  --project my-project \
  --count 5
```

### Arguments

| Flag              | Type     | Default                          | Description                                        |
| ----------------- | -------- | -------------------------------- | -------------------------------------------------- |
| `-c`, `--config`  | `Path`   | `configs/sweeps/test_sweep.yaml` | Path to the YAML sweep config file.                |
| `-p`, `--project` | `string` | `biopsy-autoencoder`             | W\&B project name under which to create the sweep. |
| `-n`, `--count`   | `int`    | `10`                             | Number of runs (trials) to execute for the sweep.  |

---

## Sweep Config File Structure

Your YAML should follow W\&B’s sweep schema. Example (`test_sweep.yaml`):

```yaml
method: bayes
metric:
  name: val_mse
  goal: minimize
parameters:
  lr:
    distribution: uniform
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [8, 16, 32]
# additional parameters…
```

`run_sweep_autoencoder.py` will add a `"device"` parameter automatically:

```yaml
parameters:
  device:
    value: "cuda:0"  # or "cpu"
```

---

## Behind the Scenes

1. **Argument Parsing**: using Python’s `argparse`.
2. **Config Validation**: checks that the provided YAML exists.
3. **W\&B Login**: prompts or reuses your API key.
4. **YAML Loading**: reads `sweep_config` via `yaml.safe_load`.
5. **Device Injection**: sets `"device": "cuda:0"` if a GPU is available.
6. **Sweep Creation**: calls `wandb.sweep(sweep_config, project=…)`, returning a `sweep_id`.
7. **Agent Launch**: runs `wandb.agent(sweep_id, function=train_sweep, count=…)`, which executes your training function `train_sweep` with each sampled config.

---

## Example

```bash
./run_sweep_autoencoder.py -c configs/sweeps/medical_autoencoder.yaml \
  -p biopsy-reconstruction -n 20
```

Output:

```
Started sweep "abc123" in project "biopsy-reconstruction". Launching 20 runs…
```

Check the W\&B dashboard to monitor progress, compare metrics, and visualize best checkpoints.

---

That's it! With `run_sweep_autoencoder.py` you can rapidly iterate on hyperparameters and track experiments end-to-end. Feel free to extend the script to include pre-sweep sanity checks or post-sweep reports.
