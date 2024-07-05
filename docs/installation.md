# Installation

This guide provides detailed instructions for setting up your development environment, configuring LLMs, and integrating various tools necessary for your project.

## Python version
We recommend using python 3.10.13

## Install with Conda
We recommend installing using Conda:
```bash
conda env create -f environment_dev.yml
conda activate AutoPrompt
```

## Install with pip
Install using pip directly:
```bash
pip install -r requirements.txt
```

## Install with pipenv
Install using pipenv:
```bash
pip install pipenv
pipenv sync
```

### Configure your LLM

Set your gateway information in the config.py.
