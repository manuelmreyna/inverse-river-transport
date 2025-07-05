# Parameter Estimation for Solute Transport in Rivers

This repository contains codes and data for parameter estimation of river transport model parameters using breakthrough curve data. It focuses on 1D immobile phase exchange models with the exchange being described by a memory function. It considers a first-order exchange memory function and a power-law memory function. It uses three different definitions of the boundary conditions. The main estimation method is DSTE-KL-PBI. This repository includes:

1. A Python implementation of the DSTE-KL-PBI method.
2. Pre-trained synthetic datasets that can be used to estimate parameters using DSTE-KL-PBI.
3. Sets of parameter estimates for 295 breakthrough curves from the TIERRAS dataset https://www.tierras.org/ (with matching labels). These estimates are obtained by DSTE-KL-PBI refined by LIPO (https://pypi.org/project/lipo/).
4. Two Jupyter notebooks that reproduce the results and figures presented in "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models", including the parameter estimation workflow and performance evaluation.

---

## Repository Structure

```
.
├── main.py                          # Main code to estimate parameters using DSTE-KL-PBI.
├── estimators/                      # Moment-based, Laplace-based, DSTE-KL-NNI and DSTE-KL-PBI estimators
├── forward/                         # Forward models (Laplace-space solution)
├── utils/                           # Utilities: loading, errors, KL, synthetic generator
├── data/                            # See below in "Data"
├── output/                          # Output of parameter estimation (CSV format)
├── methods_comparison.ipynb         # Jupyter notebook used to obtain parameters and compare models in "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models"
├── data_analysis.ipynb              # Jupyter notebook used to create plots for "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models"
└── README.md                        # This file

```

---

## Requirements

* NumPy
* SciPy
Only for methods_comparison.ipynb:
* Lipo
* PyTorch
* Scikit-Learn

---

## Data

The folder `data/` includes:
1.  `antietam_creek_tracer_data.csv`, an example of input breakthrough curve data for the model (Nordin and Sabol 1979). Breakthrough curve data can be input to the model as downloaded from the TIERRAS dataset (Rodríguez et al., 2025), the example is given in the case the format is modified.
2. `.npy` files of synthetic datasets of 10000 breakthrough curves
3. `estimated_parameters_breakthrough_curves.csv`, estimated parameters using refined DSTE-KL-PBI for 295 breakthrough curves from the TIERRAS dataset (with matching labels). 

---

## Usage of `main.py`

### 1. Estimate Parameters

Use this script to estimate parameters from real BTC data using PBI and KL decomposition.

```bash
python main.py --mode estimate_parameters
```

Other optional arguments are:

`--btcs_csv_path`, csv with the breakthrough curves to be estimated, default is 'data/antietam_creek_tracer_data.csv' which can be edited to use the same input format.  
`--data_dir`, directory with the synthetic data .npy files, default is 'data'.  
`--Nt`, number of subdivisions in a unit of dimensionless time for forward solver, default and the one that is published is 150.  
`--n_synth`, size of the synthetic dataset, default and the one that is published is 10000.   
`--seed`, seed for synthetic dataset, default and the one that is published is 1.  
`--memory_func`, memory function describing the immobile exchange, 'first order' (default) or 'power law'. 
`--bound_cond`, boundary condition and definition of the domain, 'semi-infinite-conc', 'semi-infinite-mixed' or 'infinite' (default).   
`--v_range`, range that defines the search space of the ratio between the advection velocity and the velocity of the measured peak, default is [0.9,1.5].   
`--n_vs`, number of divisions in the velocity range, default is 121.   
`--output_dir`, directory of the output files, default is 'output'.


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

`--btcs_csv_path`, csv with the breakthrough curves to be estimated, default is 'data/antietam_creek_tracer_data.csv' which can be edited to use the same input format.  
`--data_dir`, directory with the synthetic data .npy files, default is 'data'.  
`--Nt`, number of subdivisions in a unit of dimensionless time for forward solver, default and the one that is published is 150.  
`--n_synth`, size of the synthetic dataset, default and the one that is published is 10000.  
`--seed`, seed for synthetic dataset, default and the one that is published is 1.  
`--memory_func`, memory function describing the immobile exchange, 'first order' (default) or 'power law'.  
`--bound_cond`, boundary condition and definition of the domain, 'semi-infinite-conc', 'semi-infinite-mixed' or 'infinite' (default).  
`--n_lmbds`, number of terms in the KL expansion, default is 35.  

This saves:

* `params_synth_*.npy`
* `btcs_mean_*.npy`
* `phis_*.npy`
* `lambdas_*.npy`
* `Zs_*.npy`

You can control the number of breakthrough curves and seed via the script.

---

## Output Files

Each `.csv` file has rows corresponding to breakthrough curves and columns:

* Parameters: v,Pe,beta*k~_f,k~_r if first order or v,Pe,beta*alpha~,1-gamma if power law
* Errors: `RMSE` (Root Mean Square Error), `KLdiv` (Kullback–Leibler divergence)

---

## References

This work is based on "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models".

If used for publication, please cite the corresponding paper (coming soon).

---

## Authors

Developed by Manuel M. Reyna and Alexandre M. Tartakovsky, 2024–2025, as part of NSF-funded research, award number 2141503, “Collaborative Research: Informing River Corridor Transport Modeling by Harnessing Community Data and Physics-Aware Machine Learning”.

---

## Contact

For questions or collaborations, feel free to open an issue or contact via GitHub.
