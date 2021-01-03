# Lucid Torch

![Tests](https://github.com/mkirchler/feature-visualization/workflows/Tests/badge.svg)

## Setup

1. Clone the repository: `git clone https://github.com/mkirchler/feature-visualization.git`
2. Go into the folder: `cd featureVisualization`
3. Install Lucid Torch: `pip install . -f https://download.pytorch.org/whl/torch_stable.html`

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

1. [Setup](#setup) the project
2. Install `pytest` with `pip install pytest==5.3.5`
3. Run `python -m pytest`
