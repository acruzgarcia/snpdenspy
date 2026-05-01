## TODO

- [X] Optimize moment-matching fit to use the recursive approach.
- [X] Provide functionality to externally fit loc and scale (for time series analysis).
- [P] Constrained MLE where loc and scale are fixed.

- [ ] Review of the code.
- [ ] Create notebooks mimicking the examples.
- [ ] Automatic documentation.
- [ ] Readme explanation with all mathematical details.
- [ ] document all the "public" API of the library. 

## NEXT RELEASE
- [ ] provide analytical gradients in moment-matching approach.
- [ ] ? Change formaulation of snp model to be based in polinomials in monomial basis.
- [ ] ? Generalize MLE and MM fits to receive general logpdf func (maybe at cost of missing analytical gradient ?).
- [ ] Add Anderson-Darling statistic (to check GOF) and check its use in permutastion tests.
