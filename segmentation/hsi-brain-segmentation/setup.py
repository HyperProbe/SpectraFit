#!/usr/bin/env python3
"""
Setup script for HSI Brain Segmentation project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

# Remove duplicate PyYAML entries
requirements = list(dict.fromkeys(requirements))

setup(
    name="hsi-brain-segmentation",
    version="0.1.0",
    author="Tim Mach",
    author_email="tim.mach@tum.de",
    description="A comprehensive deep learning project for brain segmentation using hyperspectral imaging (HSI) data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimMachTUM/hsi-brain-segmentation",
    project_urls={
        "Bug Tracker": "https://github.com/TimMachTUM/hsi-brain-segmentation/issues",
        "Documentation": "https://github.com/TimMachTUM/hsi-brain-segmentation/blob/main/README.md",
        "Source Code": "https://github.com/TimMachTUM/hsi-brain-segmentation",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hsi-brain-seg=src.util.segmentation_util:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "hyperspectral imaging",
        "brain segmentation",
        "deep learning",
        "domain adaptation",
        "medical imaging",
        "computer vision",
        "pytorch",
        "segmentation",
    ],
    zip_safe=False,
)
