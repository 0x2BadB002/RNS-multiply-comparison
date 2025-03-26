def compute_terms(moduli):
    M = 1
    for m in moduli:
        M *= m

    terms = []
    for mi in moduli:
        Mi = M // mi
        inv = pow(Mi, -1, mi)
        term = (Mi * inv) % M
        terms.append(term)
    return terms


moduli = [2, 3, 5, 7, 11, 13, 17, 19]
terms = compute_terms(moduli)
print(terms)
