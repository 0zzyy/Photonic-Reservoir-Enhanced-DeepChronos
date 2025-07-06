# Photonic RC Chronos

This repository couples a differentiable photonic reservoir computer with the DeepChronos transformer for real-time forecasting and telecom equalisation.

## Features
- Single-node delay-based reservoir using the Ikeda map
- Optional 2‑D waveguide-mesh reservoir
- JAX custom layers integrating with DeepChronos
- Noise and hardware-aware training
- Benchmark suite: M4, Mackey–Glass, telecom channel equalisation, MIT‑BIH ECG

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart
```bash
python -m photonic_rc.benchmarks.run_m4
```

For telecom channel equalisation run the Jupyter notebook in `examples/telecom_equalisation.ipynb`.

## Repository Structure
```
photonic_rc_chronos/
  photonic_rc/        # Source code
  tests/              # Unit tests
  docs/               # Documentation
  examples/           # Example notebooks
```

## Stress Testing
The project includes a suite of 10 stress tests run via `pytest` to ensure stability under diverse conditions.

## License
Apache-2.0
