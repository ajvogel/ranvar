import ranvar as rv

def test_nonNegLower():
    rv1 = rv.NegativeBinomial(p=0.002560779768182425, n=0.010000000783403946).compute()
    assert rv1.lower() >= 0


def test_nonNegUpper():
    rv1 = rv.NegativeBinomial(p=0.002560779768182425, n=0.010000000783403946).compute()
    assert rv1.lower() <= 1000    
    
    
