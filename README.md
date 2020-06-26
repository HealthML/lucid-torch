# Lucid Torch

![Tests](https://github.com/mkirchler/feature-visualization/workflows/Tests/badge.svg)

## Setup

1. Clone the repository: `git clone https://github.com/mkirchler/feature-visualization.git`
2. Go into the folder: `cd featureVisualization`
3. Install Lucid Torch: `pip install -e .`

## Usage

Lucid Torch modules can be imported via `import lucid_torch.<module name>`. You can find usage examples here:

- [Basic](lucid_torch/examples/basic.py)
- [Multiple Objectives](lucid_torch/examples/objective.py)
- [Alpha Objective](lucid_torch/examples/alpha.py)

You can import and run the examples via

```python
from lucid_torch.examples import alpha, basic, objective

trained_alpha_image = alpha()
trained_basic_image = basic(device='cpu')
trained_objective_image = objective(numberOfFrames=500)
```

## Running the Tests

First, [setup](#setup) the project. Then run `pytest`.
