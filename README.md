# SNP Density Estimation Package (snpdenspy)

## Introduction

The `snpdenspy` package provides a Python implementation for the family of probability density functions known as semi-nonparametric (SNP). This family of densities arose from the seminal work

- *Gallant & Nychka 1987, Semi-Nonparametric Maximum Likelihood Estimation*.

and the implementation of this package is mainly based on 

- *Gallant & Tauchen 1990, A nonparametric approach to nonlinear time series analysis: estimation and simulation*. 

The package implements the theoretical framework described in the paper and does not introduce new methodological contributions. It offers efficient computational tools for density estimation, fitting, and sampling from SNP distributions.

## Mathematical Formulation

The SNP density is formulated as an expansion of the standard normal density using probabilistic Hermite polynomials. For a standardized variable $`z`$, the density is given by:

$$
f(z) = \frac{\phi(z) \left(1 + \sum_{k=1}^{K} c_k H_k(z)\right)^2}{Z}
$$

where:
- $` \phi(z) `$ is the standard normal density: $` \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2} `$
- $` H_k(z) `$ are the probabilistic Hermite polynomials (orthogonal with respect to $` \phi(z) `$)
- $` c_k `$ are real-valued coefficients
- $` Z `$ is the normalizing constant ensuring the density integrates to 1: 

$$ 
Z = 1 + \sum_{k=1}^{K} c_k^2 (k+1)! 
$$

This formulation allows for flexible density shapes and efficient evaluation while maintaining analytical tractability.

## Advantages of SNP Distributions

SNP distributions offer several key benefits:

- **Semiparametric Nature**: Combines parametric simplicity with nonparametric flexibility, allowing for rich density shapes without overfitting.
- **Analytical Properties**: Closed-form expressions for PDF, CDF, and moments.
- **Computational Efficiency**: Fast evaluation and sampling compared to fully nonparametric methods.
- **Interpretability**: The set of coefficients can be easily transformed to the to ones defining the the Edgeworth-Sargan density, which have a clear statistical interpretations related to density deviations from normality.
- **Integration**: Seamless integration with SciPy's ecosystem (especially `SNPSimpleDistribution`)


# Implemented Classes

## SNPSimpleDistribution

This class inherits from `scipy.stats.rv_continuous`, providing seamless integration with SciPy's statistical distributions framework. It supports exactly 5 Hermite coefficients ($ c_1 $ to $ c_5 $) and implements standard methods like `pdf`, `cdf`, `ppf`, and `rvs`. However, it does not include overriden fitting algorithms.

Key features:
- Inherits from `rv_continuous` for compatibility
- Fixed to 5 coefficients
- Efficient sampling via accept-reject method
- No fitting capabilities (use `SNPDistribution` for fitting)

### Basic Usage of SNPSimpleDistribution

```python
import numpy as np
from snpdenspy.snp import SNPSimpleDistribution

# Create distribution with 5 coefficients
dist = SNPSimpleDistribution()

# Evaluate PDF
x = np.linspace(-3, 3, 100)
pdf_vals = dist.pdf(x, c1=0.1, c2=0.0, c3=0.0, c4=0.0, c5=0.0, loc=0, scale=1)

# Generate samples
samples = dist.rvs(c1=0.1, c2=0.0, c3=0.0, c4=0.0, c5=0.0, size=1000)
```

## SNPDistribution

This is the general implementation, supporting an arbitrary number of Hermite coefficients. It does not inherit from `rv_continuous` but provides similar functionality through methods like `pdf`, `cdf`, `ppf`, and `rvs`. It includes both moment-matching and maximum likelihood estimation (MLE) fitting methods.

Key features:
- Arbitrary number of coefficients ($` K `$)
- Moment-matching and MLE fitting
- Flexible parameter estimation options appropiated for time series analysis.
- Frozen distribution instances via `freeze()` method

## Sampling Method

The package implements an efficient accept-reject sampling algorithm for generating random variates from SNP distributions described in the cited paper from Gallant and Tauchen 1990. The method uses a mixture of chi-squared distributions as the envelope, which is computationally superior to sampling via inverse CDF (root-finding on the CDF function) and particularly efficient for SNP distributions with moderate to high numbers of coefficients.

## Fitting Methods

### Moment Matching

This method (not directly taken from the described sources) estimates parameters by matching empirical moments to the theoretical moments of the SNP distribution.

Usage:
```python
dist = SNPDistribution(K=5)
c_fit, loc_fit, scale_fit = dist.fit_moment_matching(data)
```

### Maximum Likelihood Estimation (MLE)

The superior fitting method, implementing the Maximum Likelihhode Approach from Gallant-Nychka. It maximizes the log-likelihood function with respect to the coefficients, location, and scale parameters.

Key options:
- `fit_loc` and `fit_scale`: Control whether to estimate location and scale parameters
- `constrain_std_moments`: Apply constraints to ensure zero mean and unit variance (different than zero location and unit scale!).
- `loc_scale_fn`: Allow fitting using externally provided mean and variance functions (useful for time series analysis)
- `loc_scale_params0`, `loc_scale_bounds`: Initial values and bounds for external parameters

Usage:
```python
dist = SNPDistribution(K=5)
c_fit, loc_fit, scale_fit = dist.fit_mle(data, fit_loc=True, fit_scale=True)
```

For advanced usage with external location/scale functions:
```python
def loc_scale_fn(params, x):
    # Custom function relating params to mean/variance
    return mean, variance

c_fit, loc_fit, scale_fit = dist.fit_mle(
    data, 
    loc_scale_fn=loc_scale_fn,
    loc_scale_params0=[0.1, 0.2],
    loc_scale_bounds=[(-1, 1), (0.01, 10)]
)
```

## Usage Examples

### Advanced Usage with SNPDistribution

```python
from snpdenspy.snp import SNPDistribution

# Create distribution with arbitrary coefficients
dist = SNPDistribution(K=10)

# Fit to data using MLE
data = np.random.normal(0, 1, 1000)  # Example data
c_fit, loc_fit, scale_fit = dist.fit_mle(data)

# Create frozen distribution
frozen_dist = dist.freeze(c_fit, loc_fit, scale_fit)

# Evaluate and sample
pdf_vals = frozen_dist.pdf(x)
samples = frozen_dist.rvs(size=1000)
```

### Fitting with External Location/Scale

```python
# Example: Fit with known relationship between parameters and moments
def loc_scale_fn(params, x):
    alpha, beta = params
    mean = alpha * np.mean(x)
    variance = beta * np.var(x)
    return mean, variance

dist = SNPDistribution(K=5)
result = dist.fit_mle(
    data,
    loc_scale_fn=loc_scale_fn,
    loc_scale_params0=[1.0, 1.0],
    return_opt_result=True
)
# Access fitted external parameters
external_params = result['external_params']
```

Enjoy!
