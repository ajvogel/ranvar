import ranvar as rv

def test_nonNeg():
    rv1 = rv.NegativeBinomial(p=0.002560779768182425, n=0.010000000783403946).compute()
    assert rv1.lower() >= 0
    
    
