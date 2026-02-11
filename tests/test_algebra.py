import ranvar as mc


def test_pow():
    x = mc.Constant(3)
    y = mc.Constant(2)
    z = x ** y
    assert z.sample() == 3**2

def test_add():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x + y
    assert z.sample() == 3

def test_sub():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x - y
    assert z.sample() == -1

def test_mul():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x * y
    assert z.sample() == 2

def test_div():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x / y
    assert z.sample() == 0.5

def test_mod():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x % y
    assert z.sample() == 1

def test_floor_div():
    x = mc.Constant(1)
    y = mc.Constant(2)
    z = x // y
    assert z.sample() == 0

def test_order_of_operations():
    a = mc.Constant(2)
    b = mc.Constant(3)
    c = mc.Constant(4)
    d = mc.Constant(5)

    z = (a + b) * (c + d)

    assert z.sample() == (2 + 3) * (4 + 5)

    e = mc.Constant(6)

    z = (((a + b) * c) + d) * e

    assert z.sample() == (((2 + 3) * 4) + 5) * 6


def test_summation():
    T = mc.Constant(10)
    N = mc.Constant(5)

    som = mc.Summation(N, T)

    assert som.sample() == 50


def test_summation2():
    T = mc.Constant(10)
    N = mc.Constant(5)
    #     (50 - 30) * 20
    som = (mc.Summation(N, T) - mc.Constant(30)) * mc.Constant(20)

    assert som.sample() == 400


def test_summation_nested():
    T = mc.Summation(5,mc.Constant(1)*2) # 10
    N = mc.Summation(5,1) # 5

    som = (mc.Summation(N, T) - mc.Constant(30)) * mc.Constant(20)

    assert som.sample() == 400
