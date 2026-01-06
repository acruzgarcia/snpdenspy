
"""
Example demonstrating Hermite polynomial usage with available packages.

This example shows how to work with Hermite polynomials using SciPy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from snpdenspy.hermite.hermite_funcs import (
    hermite_polys_eval
) 


def demonstrate_hermite_polynomials():
    """Demonstrate basic Hermite polynomial usage."""
    print("Hermite Polynomials Example")
    print("=" * 40)
    
    # Generate x values
    x = np.linspace(-3, 3, 200)
    
    # Create Hermite polynomials of different degrees
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    plt.figure(figsize=(12, 8))
    
    for i, n in enumerate(degrees):
        # Physicists' Hermite polynomials
        H_n = hermite_polys_eval(n, x)
        y = H_n[n]
        
        plt.subplot(3, 3, i+1)
        plt.plot(x, y, 'b-', linewidth=2, label=f'H_{n}(x)')
        plt.grid(True, alpha=0.3)
        plt.title(f'Hermite Polynomial H_{n}(x)')
        plt.xlabel('x')
        plt.ylabel(f'H_{n}(x)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def hermite_expansion_example():
    """Show how Hermite polynomials can be used in density expansions."""
    print("\nHermite Expansion Example")
    print("=" * 40)
    
    # Generate data from a non-standard distribution
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000) + 0.3 * np.random.normal(0, 1, 1000)**3
    
    x = np.linspace(-4, 4, 200)

    # Standard normal density
    phi = norm.pdf(x, 0, 1)
    
    # Hermite expansion coefficients (simplified example)
    # In practice, these would be estimated from data
    coeffs = [1.0, 0.0, 0.3, 0.0, 0.1]  # Example coefficients
    
    # Build Hermite expansion
    expansion = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        if coeff != 0:
            He_n = hermite_polys_eval(i, x)  # Normalized Hermite polynomial
            expansion += coeff * He_n[i] * phi
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.7, label='Data histogram')
    plt.plot(x, phi, 'r--', linewidth=2, label='Standard normal')
    plt.plot(x, expansion, 'g-', linewidth=2, label='Hermite expansion')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Hermite Polynomial Expansion for Density Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Run all examples

    demonstrate_hermite_polynomials()    
    hermite_expansion_example()
