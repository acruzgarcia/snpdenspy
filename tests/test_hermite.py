
import pytest
import numpy as np

from snpdenspy.hermite.hermite_converter import HermiteMonomialConverter
from snpdenspy.hermite.hermite_funcs import (
    hermite_poly,
    hermite_product)

def test_conversions_degree_2():
    """
    Tests the monmial-hermite conversion and matrixes checks.
    """
    H2M = np.array([
        [1, 0, -1],  
        [0, 1, 0],   
        [0, 0, 1]   
    ], dtype=float)
    M2H = np.linalg.inv(H2M)
    degree = H2M.shape[0]-1

    converter = HermiteMonomialConverter()
    assert np.allclose(H2M, converter.hermite_to_monomial_matrix(degree), atol=1e-14)
    assert np.allclose(M2H, converter.monomial_to_hermite_matrix(degree), atol=1e-14)
    
    # monomial poly, by ros
    p_monomials = np.array([
        [4, -36, 81],
        [-1, 0, 1]
    ], dtype=float)
    
    # polinomials in hermite basis
    p_hermites = np.array([
        [85, -36, 81],
        [0, 0, 1]
        ], dtype=float)
    
    for i in range(0, p_monomials.shape[0]):
        
        p_monomial = p_monomials[i,:]
        p_hermite = p_hermites[i,:]
        
        p_hermite_test = converter.monomial_to_hermite(p_monomials[i,:])
        p_monomial_test = converter.hermite_to_monomial(p_hermites[i,:])

        assert np.allclose(p_hermite, p_hermite_test, atol=1e-14)
        assert np.allclose(p_monomial, p_monomial_test, atol=1e-14)
    
def test_double_conversion():
    """ Test to check that the basis-change between monomial and hermite 
    polinomials works.
    """
    
    converter = HermiteMonomialConverter()
    poly = np.array([1, 0, 2, -1, 3])
    poly_test = converter.hermite_to_monomial(converter.monomial_to_hermite(poly))
    assert np.allclose(poly, poly_test, atol=1e-14)
    
def test_double_conversion_different_degrees():
    """ Test to check that the basis-change between monomial and hermite 
    polinomials works.
    """
    def ftest(con: HermiteMonomialConverter, poly_monomial: np.array):
        
        poly_hermite = con.monomial_to_hermite(poly_monomial)
        poly_test = con.hermite_to_monomial(poly_hermite)
        
        # Format coefficients with color and one decimal place
        assert np.allclose(poly_monomial, poly_test, atol=1e-12)

    con = HermiteMonomialConverter()
    
    np.random.seed(96) # For reproducibility
    for degree in range(0, 8):
        poly = np.array([np.random.randint(-10, +10) for _ in range(degree + 1)])
        ftest(con, poly_monomial=poly)
    
def test_product_matrixes_is_eye():
    """
    Test that the product of the basis change matrices is the identity matrix
    """

    max_degree = 10
    converter = HermiteMonomialConverter()
    
    for i in range(max_degree + 1):
        HasToBeEye = converter.monomial_to_hermite_matrix(i) @ converter.hermite_to_monomial_matrix(i)
        # Assert that the product of the basis change matrices is the identity matrix
        assert np.allclose(HasToBeEye, np.eye(HasToBeEye.shape[0]), atol=1e-10)

def test_hermite_square():
    """ Test the function hermite_squre().
    """
    converter = HermiteMonomialConverter()

    np.random.seed(96) # For reproducibility
    for degree in range(0, 8):
        
        poly_hermite = np.array([np.random.randint(-10, +10) for _ in range(degree + 1)])
        poly_hermite_squared = hermite_product(poly_hermite, poly_hermite)
        
        # Testing
        poly_monomial = converter.hermite_to_monomial(poly_hermite)
        poly_monomial_squared = np.polymul(poly_monomial[::-1], poly_monomial[::-1])[::-1]
        poly_hermite_squared_test = converter.monomial_to_hermite(poly_monomial_squared)

        # Format coefficients with color and one decimal place
        assert np.allclose(
            poly_hermite_squared, poly_hermite_squared_test, atol=1e-14)

def test_conversion_for_first_hermite_polinomials():
    """
    Test to check the conversion of the first hermite polinomials
    """

    converter = HermiteMonomialConverter()

    hermite_polinomials = {
        0: [1],
        1: [0, 1],
        2: [-1, 0, 1],
        3: [0, -3, 0, 1],
        4: [3, 0, -6, 0, 1],
        5: [0, 15, 0, -10, 0, 1],
        6: [-15, 0, 45, 0, -15, 0, 1],
        7: [0, -105, 0, 105, 0, -21, 0, 1],
        8: [105, 0, -420, 0, 210, 0, -28, 0, 1],
        9: [0, 945, 0, -1260, 0, 378, 0, -36, 0, 1],
        10: [-945, 0, 4725, 0, -3150, 0, 630, 0, -45, 0, 1]
    }
    
    for degree, pol in hermite_polinomials.items():
        
        hermite_pol = converter.monomial_to_hermite(pol)

        assert all(
            (i == degree and np.isclose(v, 1.0)) or (i != degree and np.isclose(v, 0.0))
            for i, v in enumerate(hermite_pol)
        ), f"Hermite conversion degree={degree} failed"

def test_hermite_explicit_vs_recurrence():
    """
    Test to check the two functions to get hermite polinomials in monomial basis agree.
    """

    for k in range(0, 21):
        
        p1 = hermite_poly(k, method="recurrency")    
        p2 = hermite_poly(k, method="explicit")

        assert np.allclose(p1, p2, atol=1e-14)

if __name__ == "__main__":

    GREEN = "\033[32m"
    RESET = "\033[0m"

    test_double_conversion()
    print(f"Test {test_double_conversion.__qualname__} {GREEN}passed{RESET}!")
    test_double_conversion_different_degrees()
    print(f"Test {test_double_conversion_different_degrees.__qualname__} {GREEN}passed{RESET}!")
    test_product_matrixes_is_eye()
    print(f"Test {test_product_matrixes_is_eye.__qualname__} {GREEN}passed{RESET}!")
    test_conversions_degree_2()
    print(f"Test {test_conversions_degree_2.__qualname__} {GREEN}passed{RESET}!")
    test_hermite_square()
    print(f"Test {test_hermite_square.__qualname__} {GREEN}passed{RESET}!")
    test_conversion_for_first_hermite_polinomials()
    print(f"Test {test_conversion_for_first_hermite_polinomials.__qualname__} {GREEN}passed{RESET}!")
    test_hermite_explicit_vs_recurrence()
    print(f"Test {test_hermite_explicit_vs_recurrence.__qualname__} {GREEN}passed{RESET}!")
    