# HSI-Biopsy

A Python-based toolkit for hyperspectral imaging (HSI) analysis of biopsy samples, designed to streamline data loading, preprocessing, visualization, and feature extraction for medical research.

## Overview

This project focuses on the analysis of hyperspectral imaging (HSI) data for biopsy samples, with the goal of improving diagnostic capabilities through advanced imaging techniques.

## Prerequisites

- Python 3.8 or higher
- pip
- A Unix-like shell (bash) or Windows PowerShell

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TimMachTUM/hsi-biopsy.git
   cd hsi-biopsy
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # or .\.venv\Scripts\activate   # Windows
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example environment file and set your data paths:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and provide the absolute paths to your HSI data directory and metadata CSV:

   ```dotenv
   HSI_DATA_DIR=/absolute/path/to/hsi/mat/files
   METADATA_CSV_PATH=/absolute/path/to/processed/biopsy_metadata.csv
   ```

## Data Preparation

- Place your raw HSI data (`.mat` files) in the directory pointed to by `HSI_DATA_DIR`.
- Generate the metadata CSV by running the Excel processing script:
  ```bash
  python scripts/process_excel.py
  ```
- To generate preprocessed hsi_cubes and the reference spectrum for the scattering model, see the [notebooks/example_analysis.ipynb](notebooks/example_analysis.ipynb) notebook.

## Usage

For detailed instructions on how to use the dataset, including code examples and best practices, see the [Dataset Usage Guide](docs/dataset_usage.md).

For interactive analysis examples, see [notebooks/example_analysis.ipynb](notebooks/example_analysis.ipynb).

## Testing

Run unit tests with pytest:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Contributing

Contributions, bug reports, and feature requests are welcome via GitHub issues and pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
