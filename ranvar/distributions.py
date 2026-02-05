import numpy as np
import scipy as sp
from .core import RandomVariable, UPPER, LOWER
# import statsmodels.api as sm


class Constant(RandomVariable):
    def __init__(self, value) -> None:
        self.value = value

    def pmf(self, k):
        if k == self.value:
            return 1.
        else:
            return 0

    def lowerBound(self):
        return self.value

    def upperBound(self):
        return self.value + 1
        

#---------------------------------------------------------------------------------------------------


# class Uniform(RandomVariable):
#     def __init__(self, left, right) -> None:
#         nActive = right - left + 1

#         bins = np.arange(float(left), float(right + 1))
#         print(bins)
#         wgts = np.ones_like(bins)
#         print(wgts)

#         RandomVariable.__init__(self, maxBins=nActive)
#         self.setBins(bins)
#         self.setWeights(wgts)
#         self.setActiveBinCount(nActive)


class Uniform(RandomVariable):
    def __init__(self, left, right) -> None:
        self.left  = left
        self.right = right

        self.p = 1. / ((self.right - self.left) + 1)


    def pmf(self, kk):
        if self.left <= kk <= self.right:
            return self.p
        else:
            return 0

    def sample(self, size=1):
        return np.random.randint(self.left, self.right+1)


    def lower(self):
        return self.left

    def upper(self):
        return self.right


#---------------------------------------------------------------------------------------------------


class Triangular(RandomVariable):
    def __init__(self, left, mode, right, endpoints=False):

        if endpoints:
            self.left  = left - 1
            self.right = right + 1
        else:
            self.left = left
            self.right = right

        self.mode = mode

    def pmf(self, k):

        if k <= self.left:
            return 0
        elif self.left < k <= self.mode:
            return 2*(k - self.left) / ((self.right - self.left)*(self.mode - self.left))
        elif k == self.mode:
            return 2/(self.right - self.left)
        elif self.mode < k <= self.right:
            return 2*(self.right - k) / ((self.right - self.left)*(self.right - self.mode))
        else:
            return 0

    def lower(self):
        return int(self.left)

    def upper(self):
        return int(self.right)



#---------------------------------------------------------------------------------------------------

class NegativeBinomial(RandomVariable):
    def __init__(self, mean=None, dispersion=None, n=None, p=None):
        if mean is not None:
            var = mean + dispersion*mean**2
            p = mean / var
            n = mean**2 / (var - mean)            

        self.p = p
        self.n = n

    def pmf(self, k):
        return sp.stats.nbinom.pmf(k, self.n, self.p)

    @classmethod
    def fit(cls, data):
        mu  = data.mean()
        var = data.var()

        alpha = (var - mu) / mu**2

        X = np.ones_like(data)

        res = sm.NegativeBinomial(data, X).fit(
            start_params=[
                np.log(mu),
                alpha
            ]
        )

        mean = np.exp(res.params[0])
        var  = mean + res.params[1]*mean**2
        
        p = mean / var
        n = mean**2 / (var - mean)

        return cls(n=n, p=p)

    def lowerBound(self):
        return 0

    def upperBound(self):
        som = 0
        k = self.lowerBound()
        while som < UPPER:
            pk = self.pmf(k)
            som += pk
            k   += 1

        return k

        
    def toArray(self):
        outW = []
        outK = []
        som = 0
        k = self.lowerBound()
        while som < UPPER:
            pk = self.pmf(k)
            som += pk

            if som > LOWER:
                outW.append(pk)
                outK.append(k)            
            k   += 1

        outW = np.array(outW)
        outK = np.array(outK, dtype=np.intc)

        return outK, outW

    def toRanVar(self):
        k, cnts = self.toArray()
        rv = RandomVariable()
        rv.fit(k, cnts)
        return rv

if __name__ == '__main__':

    t1 = ThreePointEstimation(10,50,20)
    t1.hist()

    # dice1 = Dice()
    # bla = dice1 

    bla = t1

    for i in range(10):
        bla = bla + ThreePointEstimation(10, 50, 20)

    bla.compress()

    bla.hist()


    # for i in range(10):
    #     print(f'Iteration {i}')

    #     bla = bla + Dice()
    #     print(bla.k)
    #     print(bla.w)        
    #     print(bla.k.min())

    # bla.hist()
    # print(bla.w)
    # print(bla.k)

    # bla.compress()

    # bla.hist()
    # print(bla.w)
    # print(bla.k)    
    # print(bla.k.min())

