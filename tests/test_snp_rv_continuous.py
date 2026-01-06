import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from snpdenspy.snp.snp_rv_continuous import SNPSimpleDistribution


def test_logpdf_matches_pdf():
    dist = SNPSimpleDistribution()
    frozen = dist(c1=0.25, c2=-0.1, c3=0.05, c4=0.0, c5=0.0, loc=0.75, scale=1.1)
    x = np.linspace(-3.0, 3.0, 9)

    pdf = frozen.pdf(x)
    logpdf = frozen.logpdf(x)

    assert np.all(pdf > 0)
    assert np.allclose(logpdf, np.log(pdf), atol=1e-10)


def test_cdf_agrees_with_numerical_integration():
    frozen = SNPSimpleDistribution()(c1=0.15, c2=0.05, c3=-0.05, c4=0.0, c5=0.0, loc=-0.25, scale=1.3)
    xs = np.linspace(-2.5, 2.5, 5)

    numerical_cdf = []
    for x in xs:
        integral, error = quad(lambda t: frozen.pdf(t), -np.inf, x, limit=200)
        numerical_cdf.append(integral)

    numerical_cdf = np.asarray(numerical_cdf)
    analytical_cdf = frozen.cdf(xs)

    assert np.allclose(analytical_cdf, numerical_cdf, rtol=1e-5, atol=1e-6)


def test_moments_agree_with_numerical_integration():
    frozen = SNPSimpleDistribution()(c1=0.1, c2=-0.05, c3=0.08, c4=0.0, c5=0.0, loc=0.2, scale=1.1)
    orders = [1, 2, 3, 4]

    for n in orders:
        analytical = frozen.moment(n)
        integral, _ = quad(lambda x: (x ** n) * frozen.pdf(x), -np.inf, np.inf, limit=300)

        assert np.isfinite(analytical)
        assert analytical == pytest.approx(integral, rel=1e-5, abs=1e-6)


def test_fit_moment_matching():
    
    snp = SNPSimpleDistribution()
    true_params = dict(c1=0.2, c2=-0.1, c3=0.0, c4=0.0, c5=0.0, loc=1.5, scale=0.8)
    frozen = snp(**true_params)
    sample = frozen.rvs(size=1000, random_state=np.random.default_rng(42))

    fitted = SNPSimpleDistribution().fit(sample, method="MM")
    snp_fitted = snp(
        c1=fitted[0], c2=fitted[1], c3=fitted[2], c4=fitted[3], c5=fitted[4],
        loc=fitted[5], scale=fitted[6])

    for order in (1, 2, 3, 4):
        assert snp_fitted.moment(order) == pytest.approx(
            frozen.moment(order), rel=1e-1, abs=1e-2)


def test_pdf_integrates_to_one():
    params_list = [
        dict(c1=0.15, c2=-0.05, c3=0.02, c4=0.0, c5=0.0, loc=0.0, scale=1.0),
        dict(c1=-0.1, c2=0.08, c3=-0.04, c4=0.02, c5=0.0, loc=1.0, scale=0.9),
        dict(c1=0.05, c2=0.03, c3=0.01, c4=-0.02, c5=0.01, loc=-0.5, scale=1.2),
    ]

    for params in params_list:
        frozen = SNPSimpleDistribution()(**params)
        integral, _ = quad(lambda x: frozen.pdf(x), -np.inf, np.inf, limit=400)
        assert integral == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_pdf_matches_normal_when_shapes_zero():
    mu, sigma = 1.2, 0.7
    frozen = SNPSimpleDistribution()(c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, loc=mu, scale=sigma)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 11)

    normal_pdf = norm.pdf(x, loc=mu, scale=sigma)
    assert np.allclose(frozen.pdf(x), normal_pdf, rtol=1e-10, atol=1e-12)

if __name__ == "__main__":

    # globals for tests
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    test_logpdf_matches_pdf()
    print(f"Test {test_logpdf_matches_pdf.__qualname__} {GREEN}passed{RESET}!")
    test_pdf_integrates_to_one()
    print(f"Test {test_pdf_integrates_to_one.__qualname__} {GREEN}passed{RESET}!")
    test_cdf_agrees_with_numerical_integration()
    print(f"Test {test_cdf_agrees_with_numerical_integration.__qualname__} {GREEN}passed{RESET}!")
    test_moments_agree_with_numerical_integration()
    print(f"Test {test_moments_agree_with_numerical_integration.__qualname__} {GREEN}passed{RESET}!")
    test_fit_moment_matching()
    print(f"Test {test_fit_moment_matching.__qualname__} {GREEN}passed{RESET}!")
    test_pdf_matches_normal_when_shapes_zero()
    print(f"Test {test_pdf_matches_normal_when_shapes_zero.__qualname__} {GREEN}passed{RESET}!")
