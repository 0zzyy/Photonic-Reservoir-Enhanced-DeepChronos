# Architecture

This project combines a differentiable photonic reservoir computer with the DeepChronos transformer for sequence modelling tasks such as forecasting and telecom channel equalisation.

```
[Input] --> [Ikeda Reservoir] --> [2-D Mesh (optional)] --> [Chronos Transformer] --> [Output]
                                      |
                                      v
                              [Energy Estimator]
```

## Ikeda Reservoir
The single-node delay-based reservoir implements the Ikeda map

\begin{align}
E_{n+1} = \rho E_n e^{j(\beta |E_n|^2 + \phi)} + \gamma u_n,
\end{align}

where $E_n$ is the complex field state and $u_n$ the input at step $n$. The parameters $\rho$, $\beta$, $\phi$, and $\gamma$ control the feedback strength, Kerr non-linearity, phase offset, and input scaling respectively. This module is defined in `photonic_rc.physics.ikeda` and exposes JAX differentiable operations.

## Waveguide Mesh
An optional 2‑D waveguide mesh can be inserted after the Ikeda node to form a larger reservoir. Each step applies a phase shift $\theta$ and cyclically couples neighbouring nodes. The implementation lives in `photonic_rc.physics.mesh` and is also differentiable so it can be co-optimised with the rest of the model.

## Chronos Transformer
Reservoir states feed into a Chronos transformer backbone built on AutoGluon's `TimeSeriesPredictor`. A custom adapter in `photonic_rc.ml_layers` converts reservoir outputs to the format expected by the foundation model, enabling end-to-end training on time-series datasets.

## Energy Estimator
The `photonic_rc.utils.energy` module provides simple utilities to estimate the picojoule cost per photonic operation based on device power consumption. During experiments this estimator helps compare photonic reservoirs against purely electronic approaches.

Overall, the architecture maintains differentiability across photonic and electronic components, allowing gradient-based optimisation of reservoir parameters together with the transformer weights.
