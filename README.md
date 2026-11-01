# SpectraFit

Spectral unmixing of hyperspectral images of brain tissue: turning measured spectra into maps of oxy- and deoxyhaemoglobin, cytochrome-c-oxidase, water, fat and scattering. Developed at the Technical University of Munich within the HyperProbe project.

This repository holds two kinds of things:

- **The SpectraFit application** (this branch and the [Releases](https://github.com/HyperProbe/SpectraFit/releases)): a program with a graphical interface for experimental physicists. No programming is needed.
- **The research code** behind the HyperProbe deliverables D4.1, D4.2 and D4.3 (one branch each), as used for the papers listed below.

> **⬇ Download the SpectraFit application** for analysing your own hyperspectral measurements:
> [latest release](https://github.com/HyperProbe/SpectraFit/releases/latest) (the `SpectraFit-app` zip file).
> Unzip it and follow the user guide inside ([spectrafit/README.md](spectrafit/README.md)).

## What is where

| Where | What it contains | Paper |
|---|---|---|
| [`main`](https://github.com/HyperProbe/SpectraFit/tree/main) (this branch) | Source code and user guide of the SpectraFit application, in [`spectrafit/`](spectrafit/) | |
| [Releases](https://github.com/HyperProbe/SpectraFit/releases) | Ready-to-install versions of the application | |
| [`d4.1`](https://github.com/HyperProbe/SpectraFit/tree/d4.1) | Neural-network approach to speed up spectral unmixing: data processing, synthetic training data, networks and training scripts | [arXiv:2309.16735](https://arxiv.org/abs/2309.16735) |
| [`d4.2`](https://github.com/HyperProbe/SpectraFit/tree/d4.2) | `redefine/`: spectral unmixing methods for in-vivo brain tissue, including Monte Carlo simulations. `segmentation/`: brain segmentation with unsupervised domain adaptation | [arXiv:2503.00198](https://arxiv.org/abs/2503.00198), [arXiv:2508.16934](https://arxiv.org/abs/2508.16934) |
| [`d4.3`](https://github.com/HyperProbe/SpectraFit/tree/d4.3) | Standard spectral unmixing with the differential modified Beer–Lambert law and several optical-pathlength models, as used for the paper experiments. The application is built on this code | |
| [Zenodo](https://zenodo.org/records/23855206) | Result figures and notebooks with their plots for `d4.1` and `d4.2`, moved there to keep this repository small | |

To get the code of one branch, for example `d4.3`:

```bash
git clone --branch d4.3 https://github.com/HyperProbe/SpectraFit.git
```

Each branch has its own README with installation and usage instructions.

## Licences

- SpectraFit application (`main` and the Releases): MIT, see [LICENSE](LICENSE).
- Each research branch states its licence in its own `LICENSE` file. `d4.2` is under the PolyForm Noncommercial License 1.0.0, except for its CUDAMCML folder (GNU GPL v3).
