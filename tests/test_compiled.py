import ranvar as mc
import numpy as np

def test_isCompiled():
    rv = mc.Digest()
    rv._assertCompiled()
