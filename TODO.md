## TODO

- [X] Optimize moment-matching fit to use the recursive approach for moments.
- [ ] Constrained MLE where loc and scale are forced to be 0 and 1.
- [ ] Provide functionality to externally fit loc and scale (for time series analysis).
- [ ] ? Change formaulation of snp model to be based in polinomials in monomial basis.
- [ ] Generalize MLE and MM fits to receive general logpdf func (maybe at cost of missing analytical gradient).
- [ ] Deep review of the code.
- [ ] Create notebooks mimicking the examples.
- [ ] Automatic documentation.
- [ ] Readme explanation with all mathematical details.
- [ ] provide gradients in moment-matching approach.
- [ ] document all the "public" API of the library. 
- [ ] Add Anderson-Darling statistic (to check GOF) and check its use in permutastion tests.