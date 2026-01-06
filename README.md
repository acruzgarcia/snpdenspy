# Semiparametric Distribution Package

A Python package for working with semiparametric distributions, providing functionality similar to `scipy.stats` distributions. This package combines parametric and nonparametric components to create flexible distributions that can adapt to various data shapes.

## Project Structure

This is a **template/structure** project. The core functionality is not implemented yet - you need to implement the methods marked with `TODO` comments.

### Package Structure

```
semiparametric_dist/
├── __init__.py                 # Main package exports
├── core/
│   ├── __init__.py            # Core module exports
│   └── semiparametric_dist.py # Main SemiparametricDist class
├── fitting/
│   ├── __init__.py            # Fitting module exports
│   └── fitting.py             # Fitting methods (interfaces only)
├── sampling/
│   ├── __init__.py            # Sampling module exports
│   └── sampling.py            # Sampling methods (interfaces only)
├── visualization/
│   ├── __init__.py            # Visualization module exports
│   └── visualization.py       # Plotting functions (interfaces only)
├── tests/
│   ├── __init__.py            # Tests module
│   └── test_core.py           # Unit tests for core functionality
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
├── example.py                 # Usage examples
├── test_package.py            # Structure verification script
└── README.md                  # This file
```

## API Design

The package is designed to have the same API as `scipy.norm`:

```python
from semiparametric_dist import SemiparametricDist

# Create distribution (similar to scipy.norm)
dist = SemiparametricDist(parametric_component='normal')

# Fit to data
dist.fit(data)

# Standard statistical methods (same as scipy.norm)
pdf_values = dist.pdf(x)
cdf_values = dist.cdf(x)
samples = dist.rvs(size=1000)
mean_val = dist.mean()
std_val = dist.std()
```

## What You Need to Implement

### Core Methods in `SemiparametricDist`

1. **`_init_parametric_dist()`** - Initialize the parametric distribution component
2. **`fit(data)`** - Fit the distribution to data
3. **`pdf(x)`** - Probability density function
4. **`cdf(x)`** - Cumulative distribution function
5. **`ppf(q)`** - Percent point function (inverse CDF)
6. **`rvs(size, random_state)`** - Random variate generation
7. **`mean()`, `var()`, `std()`** - Statistical moments
8. **`median()`** - Median calculation
9. **`entropy()`** - Differential entropy
10. **`interval(confidence)`** - Confidence intervals

### Fitting Module (`fitting.py`)

1. **`fit_semiparametric()`** - Main fitting function with method selection
2. **`fit_kernel_density()`** - Kernel density estimation fitting
3. **`fit_mixture_model()`** - Gaussian mixture model fitting
4. **`fit_direct()`** - Direct parametric fitting
5. **`fit_robust()`** - Robust fitting methods
6. **`cross_validate_fit()`** - Cross-validation for model selection

### Sampling Module (`sampling.py`)

1. **`sample_semiparametric()`** - Main sampling function with method selection
2. **`sample_mcmc()`** - Markov Chain Monte Carlo sampling
3. **`sample_rejection()`** - Rejection sampling
4. **`sample_importance()`** - Importance sampling
5. **`sample_bootstrap()`** - Bootstrap resampling
6. **`sample_quantile()`** - Quantile transformation sampling
7. **`sample_mixture()`** - Mixture sampling

### Visualization Module (`visualization.py`)

1. **`plot_distribution()`** - Plot single distribution
2. **`plot_comparison()`** - Compare multiple distributions
3. **`plot_samples()`** - Plot samples with diagnostics
4. **`plot_fit_diagnostics()`** - Comprehensive diagnostic plots
5. **`plot_parameter_sensitivity()`** - Parameter sensitivity analysis
6. **`plot_confidence_intervals()`** - Confidence interval visualization

### Key Features to Implement

- **Parametric Components**: Support for normal, exponential, gamma distributions
- **Nonparametric Components**: Custom functions that modify the parametric component
- **Fitting Methods**: Maximum likelihood, method of moments, kernel density, mixture models
- **Sampling Methods**: Direct sampling, rejection sampling, MCMC, importance sampling
- **Statistical Calculations**: Moments, quantiles, entropy, confidence intervals
- **Visualization**: Distribution plots, diagnostics, comparisons, sensitivity analysis

## Installation

```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Testing

```bash
# Run structure tests
python test_package.py

# Run unit tests (once implemented)
pytest semiparametric_dist/tests/

# Run type checking
mypy semiparametric_dist/
```

## Usage Example

```python
import numpy as np
from semiparametric_dist import (
    SemiparametricDist, 
    fit_semiparametric,
    sample_semiparametric,
    plot_distribution
)

# Generate data
data = np.random.normal(0, 1, 1000)

# Create and fit distribution
dist = SemiparametricDist(parametric_component='normal')
dist.fit(data)

# Use like scipy.norm
print(f"Mean: {dist.mean():.3f}")
print(f"Std: {dist.std():.3f}")
print(f"PDF at 0: {dist.pdf(0):.3f}")

# Advanced fitting
dist_kde = fit_semiparametric(data, method='kernel', bandwidth='auto')
dist_mixture = fit_semiparametric(data, method='mixture', n_components=3)

# Advanced sampling
samples_mcmc = sample_semiparametric(dist_kde, size=1000, method='mcmc')
samples_rejection = sample_semiparametric(dist_mixture, size=1000, method='rejection')

# Visualization
plot_distribution(dist, show_data=True, show_parametric=True, show_combined=True)
```

## Development Guidelines

1. **Type Annotations**: All functions should have proper type hints
2. **Documentation**: Follow NumPy docstring format
3. **Testing**: Write comprehensive unit tests
4. **Error Handling**: Proper validation and error messages
5. **Performance**: Optimize for large datasets

## Dependencies

- **NumPy**: Numerical computing
- **SciPy**: Statistical distributions and optimization
- **Matplotlib**: Visualization
- **scikit-learn**: Machine learning algorithms (KDE, GMM)
- **pytest**: Testing framework
- **mypy**: Type checking

## Next Steps

1. Implement the `_init_parametric_dist()` method
2. Implement the `fit()` method with basic parameter estimation
3. Implement `pdf()` and `cdf()` methods
4. Add tests for each implemented method
5. Implement sampling methods
6. Add more parametric components
7. Add nonparametric component support
8. Implement fitting methods (KDE, GMM, robust)
9. Implement visualization functions
10. Optimize performance
11. Add documentation and examples

## Contributing

This is a template project. Feel free to implement the methods and contribute improvements!

## License

This project is licensed under the MIT License.
