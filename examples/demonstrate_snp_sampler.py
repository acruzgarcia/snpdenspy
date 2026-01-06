import matplotlib.pyplot as plt
import numpy as np
import time

from snpdenspy.snp.snp_distribution import SNPDistribution
from snpdenspy.util.diagnose.equal_distribution_plots import (
    qqplot_with_bootstrap_bands,
    ecdfs_with_bands
)


if __name__ == "__main__":
    # --- Create Toy distribution ---
    snp_dist = SNPDistribution(K=5)

    # Example coefficients to create a bimodal shape
    c_arr = [0.5, -0.8, 0.2, 0.0, 0.0]
    loc, scale = 12, 5
    frozen_dist = snp_dist(c=c_arr, loc=loc, scale=scale)

    # Simulate random variates and plot histogram ---
    nsamples = 5000
    
    start_time = time.time()
    samples_inv_cdf = frozen_dist.rvs(size=nsamples, method="icdf")
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f"Time taken to generate {nsamples} samples via root-finding: {elapsed:.4f} minutes")

    start_time = time.time()
    samples_accept_reject = frozen_dist.rvs(size=nsamples, method="ar")
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f"Time taken to generate {nsamples} samples via accept-reject: {elapsed:.4f} minutes")

    # Evaluate PDF on a grid and plot histogram ---
    xs = np.linspace(-28, 40, 500)
    pdf_vals = frozen_dist.pdf(xs)

    plt.figure(figsize=(8,4))
    plt.hist(samples_inv_cdf, bins=50, density=True, alpha=0.6, color="orange", label="Ppf")
    plt.hist(samples_accept_reject, bins=50, density=True, alpha=0.6, color="red", label="Accep-reject")
    plt.plot(xs, pdf_vals, label="PDF", color="blue", lw=2)
    plt.title("SNPDistribution (3 coefficients) - Random variates vs PDF")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Lets check that both set of samples comes from the same distribution

    # QQ: CI-style bands (separate resampling)
    fig1 = qqplot_with_bootstrap_bands(
        samples_inv_cdf, samples_accept_reject,
        n_boot=1000, 
        ci=0.95, 
        bootstrap_mode="ci", 
        random_state=42)[0]

    # ECDFs: bootstrap; bootstrap under H0 (pooled) so the Delta band is centered near 0
    fig2 = ecdfs_with_bands(
        samples_inv_cdf, samples_accept_reject,
        labels=("Samples with root-finder", "Samples via Accept-reject"),
        bands="bootstrap", # sDKW
        n_boot=1000,
        ci=0.95,
        bootstrap_mode="ci",  # "ci" o "null"  <---- key
        random_state=43,
        grid_mode="quantile",
        plot_diff=True,
        title="ECDFs + bands (bootstrap type: pooled)",
        )[0]

plt.show()

# TODO plot withg dsensity and density-envelope to explain the approach
# TODO Readme explain:
# - Density
# - Sampling approach
# - Hermite polinomials
