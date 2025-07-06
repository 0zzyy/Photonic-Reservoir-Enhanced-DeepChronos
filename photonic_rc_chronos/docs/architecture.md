# Architecture

```
[Input] --> [Reservoir (Ikeda)] --> [Chronos Transformer] --> [Output]
                    |
                    v
           [Energy Estimator]
```

The Ikeda reservoir follows

\begin{align}
E_{n+1} = \rho E_n e^{j(\beta |E_n|^2 + \phi)} + \gamma u_n.
\end{align}

The optional 2-D mesh applies a phase shift $\theta$ in a cyclic pattern.
