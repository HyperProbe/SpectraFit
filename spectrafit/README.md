# SpectraFit

SpectraFit turns hyperspectral images into maps of chromophore concentrations (oxy- and deoxyhaemoglobin, cytochrome-c-oxidase, water, fat, and any absorber you add) using the modified Beer–Lambert law. It has a graphical interface built on the [napari](https://napari.org) image viewer, and every analysis is saved as a small settings file so it can be repeated exactly.

It is the user-facing version of the fitting code in this repository (branch `d4.3`), reorganised so that no programming is needed.

---

## 1. Installing (macOS and Linux)

1. Download the `SpectraFit-app` zip file from the [latest release](https://github.com/HyperProbe/SpectraFit/releases/latest) and unzip it anywhere (e.g. your Downloads folder).
2. Open a Terminal in that folder. On macOS: right-click the folder in Finder → *Services* → *New Terminal at Folder*, or open Terminal, type `cd ` (with a space) and drag the folder into the window, then press Enter.
3. Type `bash install.sh` and press Enter.

It takes a few minutes and needs an internet connection. Everything goes into `~/.spectrafit` in your home folder; nothing else on the computer is changed. Afterwards you have:

* **macOS:** a `SpectraFit.command` file on your Desktop. Double-click it. A Terminal window opens alongside the program; leave it open, it shows messages.
* **Linux:** *SpectraFit* in your applications menu (and on the Desktop).
* The `spectrafit` command in the Terminal (if `~/.local/bin` is on your PATH).

To remove it: `bash uninstall.sh` in the same folder.

<details><summary>Alternative for people who already use conda or pip</summary>

```bash
pip install "./spectrafit[gui]"      # from the repository root, or "." inside this folder
spectrafit                            # opens the program
```
SpectraFit also appears in napari under *Plugins → SpectraFit*.
</details>

---

## 2. Try it with the example first

```bash
spectrafit example ~/spectrafit_example
```

This writes a synthetic 128×128 image (`example_cube.mat`, 385–1015 nm in 5 nm steps), the true maps used to make it, and matching settings. In the program, click **Load settings…**, choose `example_settings.yaml`, then **Load data** (tab 1) and **Run fit** (tab 5). Your maps should match `example_truth.npz`. You will also see warnings that water and fat cannot be determined between 530 and 750 nm (see section 6).

---

## 3. Using the program

The panel on the right has five tabs, used in order. Click any pixel in the image at any time: the **Fit inspector** plots its spectrum, or after a fit the measured and fitted attenuation with each chromophore's contribution.

### Tab 1 · Data

| Setting | What to choose |
|---|---|
| Data | A `.mat` file, or an ENVI file (a data file plus its `.hdr` header, as in the HELICOID database: choose `raw` or `raw.hdr`, or simply the sample folder). For a time series: a folder with one `.mat` file per frame, or a single file with a 4-D array. |
| Cube variable | Detected automatically (`Ref_hyper` is preferred, as written by the HyperProbe system). |
| Axis order | Detected automatically. H = rows, W = columns, B = bands, T = time, in MATLAB order. |
| Wavelengths | Read from a variable in the file when present. Otherwise 385–1015 nm / 5 nm is assumed for 127 bands, or enter the first wavelength and step. |
| Values are | *Reflectance* if already calibrated; *Raw counts* to calibrate with white and dark references: R = (raw − dark)/(white − dark). References can be variables in the same file, other `.mat` files or ENVI files; a single line or a single spectrum is fine. For ENVI data with `whiteReference`/`darkReference` files next to it (HELICOID), *Raw counts* and both references are filled in automatically. |
| Offset added to R | *Automatic* adds 0.1 to calibrated raw data (as the HELICOID pipeline does) and nothing to reflectance data. Untick to choose. |
| Binning | Averages n×n pixels while reading. 4 turns a 2048×2048 image into 512×512: 16× faster and less noisy. |
| Crop | Click *Draw…*, drag a rectangle on the image, then *Crop to drawn rectangle*. |

Then press **Load data**. A colour preview appears; the full reflectance cube is the hidden layer *SpectraFit: reflectance* (move the band slider below the image).

### Tab 2 · Reference

The differential Beer–Lambert law fits *changes* relative to a reference spectrum.

* **Single pixel:** click *Pick on image*, then click the pixel.
* **Mean of a region:** draw a polygon; its mean spectrum is the reference.
* **Spectrum from a file:** two columns, wavelength (nm) and reflectance.
* **Earlier frame (time series):** every pixel is compared with itself in the chosen frame(s), e.g. a baseline before a stimulus.
* **No reference:** the absolute attenuation −ln R is fitted instead, with concentrations constrained to be ≥ 0 and a constant offset.

*Plot reference spectrum* shows what will be used.

### Tab 3 · Model

**Fit range** (default 530–750 nm) and wavelengths to **leave out** (e.g. `600-610, 700-705`).

**Model**, with a reference (ΔA = −ln(R / R_ref)):

| Model | Equation | Fitted per pixel |
|---|---|---|
| Linear, no scattering | ΔA = PL · M Δc | Δc |
| Linear, fixed b | ΔA = PL · (M Δc + s · Δa · (λ/λ₀)^−b) | Δc, Δa (b is fixed, the same for reference and pixels) |
| a and b free | ΔA = PL · (M Δc + s · [a (λ/λ₀)^−b − a₀ (λ/λ₀)^−b₀]) | Δc, a ∈ [0, a_max], b ∈ [0, b_max] |

Without a reference the same models are applied to A = −ln R, with concentrations ≥ 0 and an optional constant offset G. Here M holds the absorption spectra (natural log, cm⁻¹ per concentration unit) and PL is the pathlength in cm. The scattering factor is s = 1/(1 − g) = 10, and λ₀ = 500 nm (both editable under *Advanced*).

In the *a and b free* model the reference's own scattering (a₀, b₀) is needed. It is either found by the same grid search as in the paper code (every grid point fits the whole subsampled image; the pair with the lowest mean error wins) or entered by hand. The default grid of 3 steps for a only tries a₀ = 0, 33.3 and 66.7; increase *a steps* for a finer search.

**Optical pathlength:**

| Option | Pathlength used | Name in the paper code |
|---|---|---|
| Constant | the value given (1 cm = unitary) | `ones` |
| Gray-matter table | measured gray-matter curve, 450–950 nm | `pathlength_from_wavelength` |
| Table from my own file | your curve: wavelength (nm), pathlength (cm) | – |
| δ-P1 model, tissue baseline | from μa = Σ baseline · spectrum and μs′ = a (λ/500)^−b, with anisotropy g and refractive index n | `gray_matter_delta_p1` |
| Jacques model, tissue baseline | same tissue optics, Jacques model (gray matter / tumour / blood vessel / general parameters) | `gray_matter_jacques` |
| δ-P1 / Jacques, per pixel | two passes: first fit with 1 cm, then each pixel's fitted μa and μs′ give its own pathlength curve, and it is fitted again | `pixelwise_…_from_unitary` |

All options work with any chromophore set, including new chromophores. For the model-based ones every chromophore contributes its baseline to μa; unknown baselines can stay 0. *Plot pathlength* shows the curve that will be used.

**Pixels to fit:** skip pixels that are too dark (background) or too bright (specular highlights), or fit only inside polygons you draw.

**Advanced:** limits of a and b, the scattering factor and λ₀, the absolute-mode offset, and the solver. *Fast* fits the whole image at once. *Exact* fits pixel by pixel with scipy, exactly like the paper code: slow, meant for checking small regions.

### Tab 4 · Chromophores

Tick the absorbers to fit. The checks on this tab use the fit range and model chosen in tab 3, so set those first. Built in (from the [White Monte Carlo framework](https://github.com/CCaredda/White-Monte-Carlo/tree/main/spectra), 400–1000 nm): HbO2, HHb, oxCCO, redCCO, water, fat, and oxidised/reduced cytochromes b and c.

* **Baseline:** the typical tissue concentration (gray-matter values by default). It is used only by the δ-P1 and Jacques pathlength models, and for StO₂ in differential mode. You can edit it in the table.
* **± resolvable:** the smallest change the fit can resolve (one standard deviation) for noise of 0.001 in ΔA, with the current wavelength range and model. Green means small compared with the baseline; orange or red means the data will hardly constrain that chromophore.
* The plot shows the selected spectra over the fit range. The coloured dot warns when two spectra are too similar to be separated.

**Adding a new chromophore:** *Add chromophore from file…*, choose a two-column text/CSV file (wavelength in nm, value), give it a name and say what the values are:

| Choice in the dialog | `unit:` in the file | Concentration unit |
|---|---|---|
| Molar extinction coefficient, base 10, cm⁻¹ M⁻¹ (e.g. Prahl tables) | `eps10_per_M_cm` | mM |
| Molar extinction coefficient, base 10, cm⁻¹ mM⁻¹ | `eps10_per_mM_cm` | mM |
| Specific absorption coefficient, natural log, cm⁻¹ mM⁻¹ | `mua_per_mM_cm` | mM |
| Absorption coefficient of the pure substance, cm⁻¹ | `mua_per_cm` | volume fraction |

It is saved in `~/.spectrafit/chromophores` and is available from then on. You can also put files there directly, in this format:

```text
# name: ICG
# label: Indocyanine green
# unit: eps10_per_M_cm
# baseline: 0
# source: where the spectrum comes from
wavelength_nm,value
600,1234
601,1250
...
```

Settings files refer to added chromophores by their file path, so the settings also work on a colleague's computer if the file is shared alongside.

### Tab 5 · Run

Press **Run fit**. Progress is shown at the bottom, and *Stop* interrupts. **Processor cores** (1 to the number of cores minus one; default all but one, so the computer stays usable) speeds up the scattering fit and the a₀/b₀ grid search; the results are identical to one core. The linear models are already fast and always use one core. With *save after each run* ticked, every run creates a new time-stamped folder. The maps appear as layers; click a name in the list to view it, or tick *side by side* to compare several. Notes and warnings about the run appear below the list.

**Save settings…** (top) writes the current settings to a file; **Load settings…** restores them.

---

## 4. What you get

Each results folder contains:

| File | Content |
|---|---|
| `settings.yaml` | all settings of this run: reload in the program or run from the command line |
| `results.mat` | all maps as MATLAB variables (time series: H×W×T) |
| `results.npz` | the same for Python (`numpy.load`) |
| `tiff/*.tif` | one 32-bit TIFF per map, for ImageJ/Fiji |
| `summary.csv` | mean, standard deviation, median, 5th/95th percentile of every map (per frame) |
| `fit_info.json` | wavelengths used, reference, a₀/b₀, warnings, run time |
| `pathlength_used.csv` | the pathlength curve (mean over pixels for per-pixel models) |

Maps: one per chromophore (Δc, or c without reference); scattering (`a`, `b`, `delta_a`, `delta_b`), or `offset` without reference; derived `HbT`, `diffCCO` and `StO2`; `residual_rms` (fit quality); `se_*` (standard error of every fitted quantity); and `flag` (0 ok, 1 a parameter at its limit, 2 not converged). In differential mode StO₂ is computed from the baseline plus the fitted changes, and its label says so. Pixels that were not fitted are NaN.

---

## 5. Command line and batch processing

```bash
spectrafit                          # open the program
spectrafit gui my_settings.yaml     # open with settings loaded
spectrafit run my_settings.yaml     # run without the interface
spectrafit run my_settings.yaml --batch "/data/session3/*.mat"   # same settings on every file
spectrafit new-config               # write a commented settings file to edit
spectrafit chromophores             # list available chromophores
```

Paths in a settings file are relative to the file. Pixel coordinates are [row, column] in the original image (before crop and binning), so a settings file stays valid when binning changes.

---

## 6. Things to know about accuracy

* **Water and fat are barely constrained below ~750 nm.** Their absorption there is tiny, and it can be traded against the scattering term. In the example (530–750 nm, noise 0.1 %) their standard errors are 13 % and 320 % of the typical tissue content, and the program warns about exactly this. Use a range extending into the near infrared (e.g. 500–900 nm) if water and fat matter, or leave them out.
* **Check the `se_*` maps** before interpreting a chromophore map: a map whose standard error is as large as its values is noise.
* **Haemoglobins and CCO** are well determined in 530–750 nm. On the example, ΔHbO2 is recovered to within 1 % of its range.
* **Binning** reduces noise as well as run time; 4×4 is a good start for 2048×2048 images.
* **Speed** for 512×512 pixels (2048² binned 4×): linear models take under 1 s. *a and b free* takes about 70 s on one core plus about 90 s for the a₀/b₀ grid search (skip it by entering a₀, b₀); both run on several cores, which on two cores halves the time and on an 8-core laptop should cut it 4–6 times. Reading a 2 GB 2048×2048×127 file with binning takes a few seconds and about 0.4 GB of memory.

---

## 7. For developers: relation to the research code

The fitting model is the one in `src/scattering_model/scatter.py` and `scatter_optim.py`. A test checks that the forward model gives identical ΔA (`tests/test_core.py::test_matches_research_code_model`). The δ-P1 and Jacques formulas were converted from the sympy pickles into plain NumPy (`_pathlength_formulas.py`) and checked to give identical values, so sympy and pickle files are no longer needed. Differences:

* **Solver.** The free-scattering fit scans b on a grid, refines it per pixel (golden section, then Gauss–Newton), and solves the linear part exactly with bounds for every b. This gives the global optimum in b on the grid instead of scipy's local optimum from (a₀, b₀). On noisy test data its residual is never larger than scipy's. The scipy pixel-by-pixel solver is still available as *Exact*.
* **Linear modes** ("no scattering" and "fixed b") are new. They are solved for the whole image in one step.
* **Per-pixel pathlengths** run both passes automatically, instead of loading the results of an earlier run.
* **Resolution.** *Binning* averages n×n pixels; the paper code's *coarseness* kept every n-th pixel. The grid search's own subsampling is a setting (default 8, as before).
* **Normalisation offset.** The +0.1 after white/dark calibration is kept as the default for raw data, and is now a visible setting.
* **Chromophores and baselines** come from files, not code. The gray-matter baseline values are the defaults.
* **HELICOID data** (ENVI files with white/dark references) is read directly, calibrated as in the research code ((raw − dark)/(white − dark) + 0.1, non-positive values set to 0.01).
* **Not carried over:** HELICOID label maps and the automatic blood-pixel reference (pick the reference pixel on the image instead), wavelength selection, and the deep-learning models. Those remain in `src/`.

Calling SpectraFit from your own Python script: the multi-core fit starts worker processes with the "spawn" method, so put the script's code under `if __name__ == "__main__":`. Without it SpectraFit notices, uses one core and says so in the notes.

Run the tests with `pip install -e ".[gui,test]" pytest-qt` and `python -m pytest` (on a Linux server without a screen: `xvfb-run -a python -m pytest`).
