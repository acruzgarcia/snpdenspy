import pytest
from snpdenspy.snp.snp_moments import Ezr_Hn, EZr_Hn_matrix


def test_EZr_Hn_matrix_matches_scalar_evaluation():
    R, N = 7, 7
    matrix = EZr_Hn_matrix(R, N)

    for r in range(R + 1):
        for n in range(N + 1):
            scalar_value = Ezr_Hn(r, n)
            assert matrix[r, n] == pytest.approx(scalar_value, rel=1e-12, abs=1e-12)


if __name__ == "__main__":

    # globals for tests
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    test_EZr_Hn_matrix_matches_scalar_evaluation()
    print(f"Test {test_EZr_Hn_matrix_matches_scalar_evaluation.__qualname__} {GREEN}passed{RESET}!")
    