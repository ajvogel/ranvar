import ranvar as rv
import numpy as np
import matplotlib.pylab as plt


def test_probability_smoothness():

    d1 = rv.NegativeBinomial(n =  10.0, p =  0.0937297862141278)
    d2 = rv.NegativeBinomial(n =  10.0, p =  0.09170673035912016)
    d3 = rv.NegativeBinomial(n =  10.0, p =  0.10434894381478518)

    D = d1 + d2 + d3

    T1 = lambda k: rv.P(D >= k)

    kk = np.array(list(range(0, 750)))
    t1 = np.array([T1(k) for k in kk])

    ttt = np.abs(np.diff(t1))

    assert max(ttt) < 0.05


if __name__ == "__main__":
    d1 = rv.NegativeBinomial(n =  10.0, p =  0.0937297862141278)
    d2 = rv.NegativeBinomial(n =  10.0, p =  0.09170673035912016)
    d3 = rv.NegativeBinomial(n =  10.0, p =  0.10434894381478518)

    D = d1 + d2 + d3

    T1 = lambda k: rv.P(D >= k)

    kk = np.array(list(range(0, 750)))
    t1 = np.array([T1(k) for k in kk])

    ttt = np.abs(np.diff(t1))

    fig = plt.figure(figsize=(16,9), dpi=100)

    ax = fig.gca()

    DD = D.compute()

    lwr = DD.lower()
    upr = DD.upper()
    print(DD.lower(), DD.upper())

    print(DD._digest.getBins())
    print(DD._digest.getWeights())

    #ax.plot(kk, rr, label='Reward')
    ax.plot(kk, t1, label='Stock Out Cost')
    # ax.plot((lwr, lwr), (0,1), color='black')
    # ax.plot((upr, upr), (0,1), color='black')
    #ax.plot(kk, t2, label='Holding Cost')


    plt.xlabel('K')
    plt.ylabel('R(k)')
    plt.legend()
    plt.tight_layout()
    fig.savefig('Reward_Function_Test.png')


    # fig2 = plt.figure(figsize=(16,9), dpi=100)
    # yy = np.array([DD.cdf(k) for k in kk])

    # ax2 = fig2.gca()
    # ax2.plot(kk, yy)

    # plt.tight_layout()
    # fig2.savefig('CDF.png')



