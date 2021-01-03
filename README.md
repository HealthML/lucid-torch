# Lucid Torch

![Tests](https://github.com/mkirchler/feature-visualization/workflows/Tests/badge.svg)

## Setup

1. Clone the repository: `git clone https://github.com/mkirchler/feature-visualization.git`
2. Go into the folder: `cd featureVisualization`
3. Install Lucid Torch: `pip install . -f https://download.pytorch.org/whl/torch_stable.html`

## Usage

You can find usage samples here:

- [Basic](lucid_torch/examples/basic.py)
- [Multiple Objectives](lucid_torch/examples/multiple_objectives.py)
- [Alpha Objective](lucid_torch/examples/alpha.py)

In [this](examples.ipynb) IPython notebook we have executed the examples.

## Running the Tests

1. [Setup](#setup) the project
2. Install `pytest` with `pip install pytest==5.3.5`
3. Run `python -m pytest`
