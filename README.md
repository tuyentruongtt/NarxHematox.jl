# NarxHematox.jl

Julia implementation of NARX neural networks with transfer learning for predicting chemotherapy-induced thrombotoxicity.

This is a port of the Python repository [NARX-Hematox](https://github.com/mariesteinacker/NARX-Hematox), which accompanies the publication "Predicting chemotherapy-induced thrombotoxicity by NARX neural networks and transfer learning" by M. Steinacker, Y. Kheifetz and M. Scholz (preprint: https://doi.org/10.1101/2024.08.06.606816).

## Installation

```julia
using Pkg
Pkg.add("NarxHematox")
```

## Usage

The notebook [`examples/NarxShowcase.ipynb`](examples/NarxShowcase.ipynb) demonstrates how individual patient dynamics are learned with NARX neural networks, and how transfer learning via a semi-mechanistic model is used. This is shown with a simulated patient time series generated with the Friberg model (Friberg et al., JCO. 20, 4713–4721 (2002),
https://doi.org/10.1200/jco.2002.02.140) with added noise and cumulative toxicity.

To run it yourself:

1. Clone this repository.
2. In Julia, activate and install the example environment:
   ```julia
   using Pkg
   Pkg.activate("examples")
   Pkg.instantiate()
   ```
3. Open `examples/NarxShowcase.ipynb` with a Julia kernel (e.g. in VS Code or Jupyter via IJulia) and run all cells.

## Usage with your own data

`train_patient` calibrates the Friberg model and trains a NARX model for one patient. It needs:

- `t_arr`: observation days as integers, starting at day 0.
- `y_arr`: platelet counts per litre.
- `cycle_arr`: the therapy cycle of each observation (the `CYCLE` column of the data). It is required, because cycle lengths differ between patients (for example 14 or 21 days).
- `therapy`: `(therapy days, doses)`, one therapy per cycle.

Observations with `cycle_arr .<= train_cycles` are used for training, the rest for testing.

```julia
using NarxHematox
res = train_patient(t_arr, y_arr, cycle_arr, therapy; celltype = "GRU", train_cycles = 4)
```

`res.y_narx` and `res.y_fri` are the daily predictions, and `res.smse_narx` and `res.smse_fri` are the test errors.

## File description

- `src/friberg.jl`: Friberg model, its calibration loss and helpers.
- `src/narx.jl`: ARX-RNN (with a TensorFlow-compatible GRU cell) and ARX-FNN models.
- `src/utils.jl`: losses, regularization, pruning, early stopping and model builders.
- `src/weights.jl`: loading of the original Keras `.h5` weights.
- `src/fit.jl`: training loop.
- `src/training.jl`: Friberg calibration and the three-phase NARX training for one patient (`calibrate_friberg`, `train_patient`).
- `examples/NarxShowcase.ipynb`: demonstration notebook.
- `data/therapies_all.csv`: generated treatment scenarios for transfer learning, as used in Steinacker et al., Heliyon. 9, e17890 (2023), https://doi.org/10.1016/j.heliyon.2023.e17890.
- `data/index_weights_GRU.h5`, `data/index_weights_FNN.h5`: starting weights for transfer learning, from the original Python repository.
- `test/runtests.jl`: tests, including checks against the Python implementation.

## Differences from the Python version

- The Friberg model is solved with `Tsit5` (OrdinaryDiffEq.jl) instead of `LSODA`; results agree to about four digits.
- Friberg calibration uses Nelder-Mead from Optim.jl instead of SciPy.
- Random numbers differ, so simulated noise and training results differ slightly.
- Dropout and GRU activations other than sigmoid are not implemented; the original showcase does not use them.
- Missing observations (`NaN`) are skipped in the Friberg calibration loss. In Python they make the loss `NaN`, and the calibration silently returns the population parameters. With many missing observations, the calibrated parameters become less reliable.
- If the first training observation is missing, the warm-up value is the calibrated baseline $c_0$. The Python notebook uses $c_0/10^9$ here, without the factor $10^9$.

## Citation

If you use this package, please cite the original publication (see above) and this software: https://doi.org/10.5281/zenodo.23498911.

## License

AGPL-3.0, as the original repository. See [`LICENSE`](LICENSE).

## AI assistance

This Julia port was developed with the help of Claude Opus 5.5 (Anthropic), an AI assistant, in October 2026. Claude assisted with reviewing the translation from the Python code to Julia, fixing errors, and writing tests.
All code was finally reviewed, run and tested by the author, who takes full responsibility for its content.