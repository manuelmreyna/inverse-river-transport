# Parameter Estimation for Solute Transport in Rivers

This repository contains two main workflows:

1. **Parameter Estimation Using Projected Barycentric Interpolation (and Nearest Neighbor Interpolation) on an existing synthetic dataset**
2. **Synthetic Data Generation**
 
These tools are designed for studying solute transport with mobile-immobile exchange in rivers, using BTC (Breakthrough Curve) data, Laplace/Moment estimators are used to obtain approximate estimates, a Laplace domain forward solution is used, and Karhunen–Loève decomposition with Projected Barycentric Interpolation (PBI) is used for the final parameter estimation.

---

## 📁 Repository Structure

```
.
├── main.py                          # Generate synthetic BTC dataset and save KL decomposition
├── estimators/                      # Moment-based, Laplace-based, and KL-PBI estimators
├── forward/                         # Forward models (Laplace-space solution)
├── utils/                           # Utilities: loading, errors, KL, synthetic generator
├── data/                            # Input BTCs (Antietam Creek data is given to show format of input, Nordin and Sabol 1979) and output generated synthetic datasets
├── output/                          # Output of parameter estimation (CSV format)
└── README.md                        # This file
```

---

## ⚙️ Requirements

* Python 3.8+
* NumPy
* SciPy

---

## 🚀 Usage

### 1. Estimate Parameters

Use this script to estimate parameters from real BTC data using PBI and KL decomposition.

```bash
python main.py --mode estimate_parameters
```

Other optional arguments are:

--btcs_csv_path, csv with the breakthrough curves to be estimated, default is 'data/antietam_creek_tracer_data.csv' which can be edited to use the same input format.
--data_dir, directory with the synthetic data .npy files, default is 'data'
--Nt, number of subdivisions in a unit of dimensionless time for forward solver, default and the one that is published is 150
--n_synth, size of the synthetic dataset, default and the one that is published is 10000
--seed, seed for synthetic dataset, default and the one that is published is 1
--memory_func, memory function describing the immobile exchange, 'first order' (default) or 'power law'
--bound_cond, boundary condition and definition of the domain, 'semi-infinite-conc', 'semi-infinite-mixed' or 'infinite' (default)
--v_range, range that defines the search space of the ratio between the advection velocity and the velocity of the measured peak, default is [0.9,1.5]
--n_vs, number of divisions in the velocity range, default is 121
--output_dir, directory of the output files, default is 'output'


This saves:

* `params_NNI.csv`: Parameters estimated using Nearest Neighbor Interpolation
* `params_PBI.csv`: Parameters estimated using Projected Barycentric Interpolation
* `errors_NNI.csv`, `errors_PBI.csv`: Errors (RMSE, KLdiv) for each BTC

---

### 2. Generate Synthetic Data

This creates synthetic BTCs from distributions inferred from field BTCs.

```bash
python main.py --mode generate_synthetic
```

Other optional arguments are:

--btcs_csv_path, csv with the breakthrough curves to be estimated, default is 'data/antietam_creek_tracer_data.csv' which can be edited to use the same input format.
--data_dir, directory with the synthetic data .npy files, default is 'data'
--Nt, number of subdivisions in a unit of dimensionless time for forward solver, default and the one that is published is 150
--n_synth, size of the synthetic dataset, default and the one that is published is 10000
--seed, seed for synthetic dataset, default and the one that is published is 1
--memory_func, memory function describing the immobile exchange, 'first order' (default) or 'power law'
--bound_cond, boundary condition and definition of the domain, 'semi-infinite-conc', 'semi-infinite-mixed' or 'infinite' (default)
--n_lmbds, number of terms in the KL expansion, default is 35

This saves:

* `params_synth_*.npy`
* `btcs_mean_*.npy`
* `phis_*.npy`
* `lambdas_*.npy`
* `Zs_*.npy`

You can control the number of BTCs and seed via the script.

---

## 📊 Output Files

Each `.csv` file has rows corresponding to BTCs and columns:

* Parameters: v,Pe,beta*k~_f,k~_r if first order or v,Pe,beta*alpha~,1-gamma if power law
* Errors: `RMSE`, `KLdiv`

---

## 📚 References

This work is based on:

* Reduced Order Modeling using Karhunen–Loève decomposition
* Inverse modeling of advection-dispersion equations with mobile-immobile mass exchange
* Projected Barycentric Interpolation (PBI)

If used for publication, please cite the corresponding papers (coming soon).

---

## 🧑‍💻 Authors

Developed by Manuel M. Reyna and Alexandre M. Tartakovsky, 2024–2025, as part of NSF-funded research, award number 2141503, “Collaborative Research: Informing River Corridor Transport Modeling by Harnessing Community Data and Physics-Aware Machine Learning”.

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact via GitHub.
