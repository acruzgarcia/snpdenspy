import numpy as np
from .hermite_funcs import hermite_to_monomial_matrix

class HermiteMonomialConverter:
    """# Singleton implementation to compute monomial-hermite (and vice versa)
    conversions of polinomials
    """
    _instance = None  # Singleton instance
    _max_degree = 20

    def __new__(cls):
        """
        Singleton initialization. It initializes the matrices once up to max_degree.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_matrices(max_degree=cls._max_degree)
        return cls._instance

    def _init_matrices(self, max_degree: int):
        """Precompute the change-of-basis matrices once up to max_degree, and 
        store it in the class."""
        self.max_degree = max_degree

        # Store all the chage-of-matrixes and its inverses to avoid mistakes
        # TODO: We should be able to  simply slide the max_degree change-of-matrix
        # to compute the lower degress
        self._H2M = dict()
        self._inverses_H2M = dict()
        
        for degree in range(0, self.max_degree + 1):
            self._H2M[degree] = hermite_to_monomial_matrix(degree)
            self._inverses_H2M[degree] = np.linalg.inv(self._H2M[degree])

    def hermite_to_monomial_matrix(self, degree: int) -> np.ndarray:
        """It returns the change-of-basis matrix M (hermnite-to-monomial) corresponding to the degree 
        of the polynomial.
        """
        return self._H2M[degree]

    def monomial_to_hermite_matrix(self, degree: int) -> np.ndarray:
        """It returns the change-of-basis matrix M (monomial-to-hermite) corresponding to the degree 
        of the polynomial.
        """
        return self._inverses_H2M[degree]
    
    def hermite_to_monomial(self, poly_hermite: np.ndarray) -> np.ndarray:
        """It converts the input polinomial, which is expressed in hermite basis, 
        into a polinomial expressed in the monomial basis.
        """    
        mat = self.hermite_to_monomial_matrix(len(poly_hermite)-1)
        return mat @ poly_hermite

    def monomial_to_hermite(self, poly: np.ndarray) -> np.ndarray:
        """It converts the input polinomial, which is expressed in monomial basis, 
        into a polinomial expressed in the hermitian basis.
        """    
        mat = self.monomial_to_hermite_matrix(len(poly)-1)
        return mat @ poly

# End class
