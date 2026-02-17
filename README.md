# Option Pricing — Monte Carlo and PDE Methods  
### Numerical Comparison under the Black–Scholes Framework

---

## Overview

This project implements a structured comparison of numerical methods for European option pricing under the Black–Scholes model.

The objective is to:

- Implement the closed-form Black–Scholes solution as a benchmark  
- Develop a Monte Carlo pricing framework  
- Implement finite-difference PDE schemes  
- Compare methods in terms of accuracy, stability and convergence  
- Analyze computational trade-offs between probabilistic and deterministic approaches  

Full HTML notebook available here: 
(https://Morales-Mathieu.github.io/option-pricing-monte-carlo-pde/index.html)

---

## Methods Implemented

### 1. Closed-Form Black–Scholes Pricing
- Analytical benchmark solution  
- Used to measure numerical error  
- Reference for convergence analysis  

### 2. Monte Carlo Simulation
- Plain Monte Carlo estimator  
- Antithetic variates (variance reduction)  
- Empirical verification of statistical convergence  
- Error analysis as a function of simulation size  

### 3. Finite-Difference Schemes (PDE)

#### Explicit Scheme
- Forward time discretization  
- Stability constraint analysis  
- Sensitivity to grid parameters  

#### Implicit Scheme
- Backward time discretization  
- Unconditional stability  
- Improved robustness under grid refinement  

#### Crank–Nicolson Scheme
- Semi-implicit method  
- Higher numerical accuracy  
- Balanced stability and precision  

---

## Comparative Framework

The methods are compared using:

- Absolute pricing error relative to the analytical benchmark  
- Convergence behavior under grid refinement  
- Stability properties  
- Computational cost considerations  

The project highlights the trade-offs between:

- Statistical convergence (Monte Carlo)  
- Deterministic convergence (PDE methods)  
- Flexibility versus numerical efficiency  

---

## Project Structure

```bash
option-pricing-monte-carlo-pde/
│
├── notebooks/
│   └── option_pricing_mc_vs_pde.ipynb
│
├── src/
│   ├── bs.py
│   ├── mc.py
│   ├── pde.py
│   └── plots.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```
## How to run 

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```


## Run the Main Notebook

Open and execute:
```bash
    notebooks/option_pricing_mc_vs_pde.ipynb
```
