import ranvar as mc
import numpy as np
import sys

# def test_addTwoUniform():
#     """addUniform"""
#     x = mc.Uniform(1,6) + mc.Uniform(1,6)
#     # print(x.lower)
#     # print(x.upper)
#     # print(x.count)
#     # print(x.known)
#     # assert x._assertConnected()

#     # print(x.nActive)
#     # print(x.maxBins)

#     for i in range(2, 13):
#         print(i, x.pmf(i))

#     print("Bins = ",x.getBins())
#     print("Wgts = ",x.getWeights())

#     assert abs(x.pmf(2) - 0.0277) <= 1e-4
#     assert abs(x.pmf(3) - 0.0555) <= 1e-4
#     assert abs(x.pmf(4) - 0.0833) <= 1e-4
#     assert abs(x.pmf(5) - 0.1111) <= 1e-4
#     assert abs(x.pmf(6) - 0.1388) <= 1e-4
#     assert abs(x.pmf(7) - 0.1666) <= 1e-4
#     assert abs(x.pmf(8) - 0.1388) <= 1e-4
#     assert abs(x.pmf(9) - 0.1111) <= 1e-4
#     assert abs(x.pmf(10) - 0.0833) <= 1e-4
#     assert abs(x.pmf(11) - 0.0555) <= 1e-4
#     assert abs(x.pmf(12) - 0.0277) <= 1e-4


# def test_lowerBound_and_upperBound():
#     x = mc.Uniform(1,6) + mc.Uniform(1,6)
#     assert x.lower() == 2
#     assert x.upper() == 12

# def test_nActiveCount():
#     """Tests that the frequency column adds up to the number of points added"""
#     std = 100
#     mu  = 100
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = mc.RandomVariable(maxBins=16)
#     for d in data:
#         x.add(d)

#     # print(x.nAc-tive)

#     assert x.getActiveBinCount() == 16

# NO LONGER APPLICABLE
# def test_binConnectivity():
#     """Checks that bins don't become disconnected."""
#     np.random.seed(300)
#     std = 100
#     mu  = 1000
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = mc.RandomVariable(maxBins=16)
#     for e, d in enumerate(data):
#         # print()
#         print(f'Adding {d} as point {e+1}...')
#         x.add(d)
#         if e >= 16:
#             x._assertConnected()
#             # assert False

#     x._assertConnected()

# def test_freqAddsUp():
#     """Tests that the frequency column adds up to the number of points added"""
#     std = 100
#     mu  = 100
#     N   = 100
#     data = np.random.randn(N)*std + mu
#     x = mc.RandomVariable(maxBins=16)
#     for d in data:
#         x.add(d)

    # print(sum(x.freq))

    # assert (sum(x.getWeights()) == N)


# def test_normalApprox():
#     """"""
#     std = 100
#     mu  = 100
#     np.random.seed(31337)
#     data = np.random.randn(10000)*std + mu
#     x = mc.RandomVariable(maxBins=32)
#     for d in data:
#         x.add(d)


#     prob1 = x.cdf(mu + std*1) - x.cdf(mu - std*1)
#     prob2 = x.cdf(mu + std*2) - x.cdf(mu - std*2)
#     prob3 = x.cdf(mu + std*3) - x.cdf(mu - std*3)

#     # We are not using a lot of samples and keeping the accuracy bar low, otherwise
#     # running the tests will take to long. In practice increasing the number of
#     # sample points will increase the accuracy.

#     print(prob1)
#     print(prob2)
#     print(prob3)

#     assert abs(0.6827 - prob1) <= 1e-1
#     assert abs(0.9545 - prob2) <= 1e-2
#     assert abs(0.9973 - prob3) <= 1e-3

# def test_normalApprox_quantile():
#     """"""
#     std = 100
#     mu  = 100
#     np.random.seed(31337)
#     data = np.random.randn(10000)*std + mu
#     x = mc.RandomVariable(maxBins=64)
#     for d in data:
#         x.add(d)

#     DATA = [
#         (-64.485, 0.05),
#         (32.551, 0.25),
#         (100.00, 0.50),
#         (167.449, 0.75),
#         (264.485, 0.95)
#     ]

#     for v, p in DATA:
#         dv = abs((v - x.quantile(p)) / v)
#         print(f'{v} : {x.quantile(p)} : {dv}')
#         assert dv <= 1e-1

    #assert False




def test_randInt_sample():
    dice = mc.RandInt(1,6).sample()
    assert 1 <= dice <= 6

# def test_randInt_compute():
#     dice = mc.RandInt(1,6).compute()

#     assert dice.getActiveBinCount() == 6
#     print(dice)
#     print(dice.getBins())
#     print(dice.getWeights())

#     assert False

def test_randInt_distribution():
    dice = mc.RandInt(1,6).compute()._digest
    nActive = dice.getActiveBinCount()
    minW = min(dice.getWeights()[:nActive])
    maxW = max(dice.getWeights()[:nActive])

    avgW = (maxW + minW) / 2

    assert nActive == 6

    print(dice.getWeights())

    print(minW)
    print(maxW)
    print(maxW - minW)
    print((maxW - minW) / avgW)

    assert abs(maxW - minW) / avgW <= 1e-1




def test_sumDices():
    data6 = [
        [6,0.00214334705075],
        [7,0.0128600823045],
        [8,0.0450102880658],
        [9,0.120027434842],
        [10,0.270061728395],
        [11,0.54012345679],
        [12,0.977366255144],
        [13,1.62037037037],
        [14,2.48842592593],
        [15,3.57081618656],
        [16,4.81610082305],
        [17,6.12139917695],
        [18,7.35382373114],
        [19,8.37191358025],
        [20,9.04706790123],
        [21,9.28497942387],
        [22,9.04706790123],
        [23,8.37191358025],
        [24,7.35382373114],
        [25,6.12139917695],
        [26,4.81610082305],
        [27,3.57081618656],
        [28,2.48842592593],
        [29,1.62037037037],
        [30,0.977366255144],
        [31,0.54012345679],
        [32,0.270061728395],
        [33,0.120027434842],
        [34,0.0450102880658],
        [35,0.0128600823045],
        [36,0.00214334705075]
        ]

    out = [mc.RandInt(1,6) for ii in range(6)]
    out2 = out[0] + out[1] + out[2] + out[3] + out[4] + out[5]
    out2 = out2.compute(samples=10_000)
    #out2 = mc.SUM(5, mc.RandInt(1,6)).compute()

    # cnts = out2.getCountArray()
    # print(sum(cnts))
    #
    yActual = np.array([d[1]/100 for d in data6]).cumsum()
    yTest   = np.array([out2.cdf(k) for k in range(6,37)])

    with open('output.csv','w') as fout:
        fout.write('Actual,Estimate\n')
        for a, e in zip(yActual, yTest):
            fout.write(f'{a*100:3.3f},{e*100:3.3f}\n')

    error   =  (((yTest - yActual)**2).mean())**0.5

    assert error < 5.e-2
    print(yActual)
    print(yTest)
    print(error)




def test_quantileSample():
    x = [
        -3719.0165,        -2322.6865,        -2051.7702,        -1879.4139,        -1749.6195,        -1643.9816,
        -1554.0353,        -1475.1508,        -1404.5068,        -1340.2502,        -1281.0959,        -1226.1134,
        -1174.6070,        -1126.0414,        -1079.9959,        -1036.1332,        -994.1784,        -953.9045,
        -915.1212,        -877.6678,        -841.4069,        -806.2200,        -772.0041,        -738.6690,
        -706.1353,        -674.3324,        -643.1974,        -612.6739,        -582.7108,        -553.2620,
        -524.2855,        -495.7426,        -467.5981,        -439.8193,        -412.3758,        -385.2395,
        -358.3840,        -331.7845,        -305.4178,        -279.2617,        -253.2953,        -227.4987,
        -201.8525,        -176.3385,        -150.9388,        -125.6361,        -100.4136,        -75.2548,
        -50.1435,        -25.0639,        0.0000,        25.0639,        50.1435,        75.2548,        100.4136,
        125.6361,        150.9388,        176.3385,        201.8525,        227.4987,        253.2953,        279.2617,
        305.4178,        331.7845,        358.3840,        385.2395,        412.3758,        439.8193,        467.5981,
        495.7426,        524.2855,        553.2620,        582.7108,        612.6739,        643.1974,        674.3324,
        706.1353,        738.6690,        772.0041,        806.2200,        841.4069,        877.6678,        915.1212,
        953.9045,        994.1784,        1036.1332,        1079.9959,        1126.0414,        1174.6070,        1226.1134,
        1281.0959,        1340.2502,        1404.5068,        1475.1508,        1554.0353,        1643.9816,        1749.6195,
        1879.4139,        2051.7702,        2322.6865,        3719.0165
    ]

    mu  = 0.0
    std = 1000.0
    rv = mc.Quantiles(*x).compute()

    prob1 = rv.cdf(mu + std*1) - rv.cdf(mu - std*1)
    prob2 = rv.cdf(mu + std*2) - rv.cdf(mu - std*2)
    prob3 = rv.cdf(mu + std*3) - rv.cdf(mu - std*3)

    # We are not using a lot of samples and keeping the accuracy bar low, otherwise
    # running the tests will take to long. In practice increasing the number of
    # sample points will increase the accuracy.

    print(prob1)
    print(prob2)
    print(prob3)

    assert abs(0.6827 - prob1) <= 1e-1
    assert abs(0.9545 - prob2) <= 1e-2
    assert abs(0.9973 - prob3) <= 1e-2


def test_array_sum():



    array = [mc.Constant(i) for i in range(10)]
    array2 = [i for i in range(10)]
    start = 0
    end = 10

    assert mc.ArraySum(array, 0, 10).sample() == sum(array2)
    assert mc.ArraySum(array, 0, 5).sample()  == sum(array2[0:5])
    assert mc.ArraySum(array, 0, 1).sample()  == sum(array2[0:1])
    assert mc.ArraySum(array, 5, 10).sample()  == sum(array2[5:10])

    x = mc.Constant(5) + mc.ArraySum(array, 5, 10) + mc.Constant(10)


    assert x.sample()  == 5 + sum(array2[5:10]) + 10


def test_sample_digest():
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(100_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    y = mc.DigestVariable(x)
    #y = y.sample()
    y = y.compute(samples=100_000)
    print(y)
    DATA = [
        (-64.485, 0.05),
        (32.551, 0.25),
        (100.00, 0.50),
        (167.449, 0.75),
        (264.485, 0.95)
    ]

    for v, p in DATA:
        dv = abs((v - y.quantile(p)) / v)
        print(f'{v} : {x.quantile(p)} : {y.quantile(p)} : {dv}')
        assert dv <= 5e-1

    #assert False
