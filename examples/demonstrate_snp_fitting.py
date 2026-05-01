import numpy as np
from scipy import stats

from snpdenspy.snp.snp_distribution import SNPDistribution
from snpdenspy.util.diagnose.diagnose_fit import diagnose_fit


def _sample_gaussian_mixture(mus, sigmas, weights, nsamples, rng=None):
    """
    Draw samples from a univariate Gaussian mixture.
    """
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if not (len(mus) == len(sigmas) == len(weights)):
        raise ValueError("mus, sigmas, and weights must have the same length.")

    weights = weights / np.sum(weights)
    if rng is None:
        rng = np.random.default_rng()

    comps = rng.choice(len(mus), size=nsamples, p=weights)
    z = rng.standard_normal(nsamples)
    return mus[comps] + sigmas[comps] * z

if __name__ == "__main__":

    # --- Create example distribution as a Gaussian mixture ---
    mus = [-6.0, 6.0]
    sigmas = [1.0, 2.5]
    weights = [0.3, 0.7]
    rng = np.random.default_rng(98)

    # Sample of th emixture
    def sample_gaussian_mixture(nsamples, rng):
        return _sample_gaussian_mixture(
            mus=mus, sigmas=sigmas, weights=weights, nsamples=nsamples, rng=rng
        )

    # Evaluation of the pdf of the mixture
    def pdf_mixture_eval(xs):    
        def _phi(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return sum(
            weights[i] * _phi(xs, mus[i], sigmas[i]) for i in range(len(mus))
        )

    # --- Additional analytical distributions ---
    student_t_dist = stats.t(df=5)
    laplace_dist = stats.laplace(loc=0.0, scale=1.5)
    logistic_dist = stats.logistic(loc=0.0, scale=1.0)
    skewnorm_dist = stats.skewnorm(a=8.0, loc=0.0, scale=1.0)
    chi_dist = stats.chi(df=3.0)
    chi2_dist = stats.chi2(df=4.0)

    def sample_student_t(nsamples, rng):
        return student_t_dist.rvs(size=nsamples, random_state=rng)

    def pdf_student_t_eval(xs):
        return student_t_dist.pdf(xs)

    def sample_laplace(nsamples, rng):
        return laplace_dist.rvs(size=nsamples, random_state=rng)

    def pdf_laplace_eval(xs):
        return laplace_dist.pdf(xs)

    def sample_logistic(nsamples, rng):
        return logistic_dist.rvs(size=nsamples, random_state=rng)

    def pdf_logistic_eval(xs):
        return logistic_dist.pdf(xs)

    def sample_skewnorm(nsamples, rng):
        return skewnorm_dist.rvs(size=nsamples, random_state=rng)

    def sample_chi(nsamples, rng):
        return chi_dist.rvs(size=nsamples, random_state=rng)

    def pdf_chi_eval(xs):
        return chi_dist.pdf(xs)

    def sample_chi2(nsamples, rng):
        return chi2_dist.rvs(size=nsamples, random_state=rng)

    def pdf_chi2_eval(xs):
        return chi2_dist.pdf(xs)

    def pdf_skewnorm_eval(xs):
        return skewnorm_dist.pdf(xs)

    distributions = [
        ("Gaussian mixture", sample_gaussian_mixture, pdf_mixture_eval),
        ("Student's t (df=5)", sample_student_t, pdf_student_t_eval),
        ("Laplace(loc=0, scale=1.5)", sample_laplace, pdf_laplace_eval),
        ("Logistic(loc=0, scale=1)", sample_logistic, pdf_logistic_eval),
        ("Skew-normal(a=8)", sample_skewnorm, pdf_skewnorm_eval),
        ("Chi(df=3)", sample_chi, pdf_chi_eval),
        ("Chi-squared(df=4)", sample_chi2, pdf_chi2_eval),
    ]

    for name, sampler, pdf_func in distributions:
        
        K = 5
        x = sampler(nsamples=4000, rng = np.random.default_rng())
        c_hat, loc_hat, scale_hat = SNPDistribution(K=K).fit_mle(x=x, fit_loc=True, fit_scale=True)
        fitted_dist = SNPDistribution(K=K)(c=c_hat, loc=loc_hat, scale=scale_hat)

        print(f"\n=== {name} ===")
        diagnose_fit(
            x=x,
            fitted_dist=fitted_dist,
            pdf_original_func=pdf_func,
            title=name,
        )
