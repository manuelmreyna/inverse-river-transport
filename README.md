# Parameter Estimation for Solute Transport in Rivers

This repository contains codes and data for parameter estimation of river transport model parameters using breakthrough curve data. It focuses on 1D immobile phase exchange models with the exchange being described by a memory function. It considers a first-order exchange memory function and a power-law memory function. It uses three different definitions of the boundary conditions. The main estimation method is DROSTE. This repository includes:

1. A Python implementation of the DROSTE method.
2. Pre-trained synthetic datasets that can be used to estimate parameters using DROSTE.
3. Sets of parameter estimates for 295 breakthrough curves from the TIERRAS dataset https://www.tierras.org/ (with matching labels). These estimates are obtained by DROSTE refined by LIPO (https://pypi.org/project/lipo/).
4. Four Jupyter notebooks that reproduce the results and figures presented in "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models", including the parameter estimation workflow and performance evaluation.

---

## Repository Structure

```
.
├── code
|    ├── estimators/                      # Moment-based, Laplace-based estimators
|    ├── forward/                         # Forward models (Laplace-space solution)
|    └── utils/                           # Utilities: loading, errors
├── data/                            # See below in "Data"
├── output/                          # Output folder for parameter estimation (CSV format) and `estimated_parameters_breakthrough_curves_with_hydraulic_data.csv` (estimated parameters using refined DROSTE for 295 breakthrough curves from the TIERRAS dataset, with matching labels and hydraulic paramters from the TIERRAS dataset)
├── notebooks/
|    ├── compare_methods.ipynb         # Jupyter notebook used to obtain parameters and compare models in "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models" 
|    ├── data_analysis.ipynb           # Jupyter notebook used to create plots for "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models"
|    ├── fit_parameters.ipynb          # Jupyter notebook used to obtain DROSTE parameter estimates for "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models"
|    └── uncertainty_quantification.ipynb          # Jupyter notebook used to perform uncertainty quantification using DROSTE for "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models"
└── README.md                          # This file

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


## References

This work is based on "Parameter estimation in river transport models with immobile phase exchange using dimensional analysis and reduced-order models".

If used for publication, please cite the corresponding paper (coming soon).

---

## Authors

Developed by Manuel M. Reyna and Alexandre M. Tartakovsky, 2024–2025, as part of NSF-funded research, award number 2141503, “Collaborative Research: Informing River Corridor Transport Modeling by Harnessing Community Data and Physics-Aware Machine Learning”.

---

## Contact

For questions or collaborations, feel free to open an issue or contact via GitHub.
