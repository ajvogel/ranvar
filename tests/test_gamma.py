import ranvar as rv
import numpy as np


# Theoretical mean of Gamma(shape, scale, location) = location + shape * scale
# Theoretical variance = shape * scale^2


def _compute(node, samples=50_000):
    return node.compute(samples=samples)


# --[ Opcode-level VM tests ]---------------------------------------------------

def test_gamma_vm_bytecode():
    """Gamma samples directly via VM bytecode are finite and positive."""
    shape, scale = 2.0, 3.0
    BYTECODE = [
        (rv.OP_PUSH, shape),
        (rv.OP_PUSH, scale),
        (rv.OP_PUSH, 0.0),       # location = 0
        (rv.OP_RAND_GAMMA, 0),
    ]
    opcodes  = np.array([op for op, _ in BYTECODE], dtype=np.double)
    operands = np.array([oe for _, oe in BYTECODE], dtype=np.double)

    vm = rv.VirtualMachine(opcodes, operands)
    d = vm.compute(samples=20_000)

    # mean ≈ shape * scale = 6
    assert abs(d.quantile(0.5) - shape * scale) < 1.5


def test_gamma_vm_with_location():
    """Shifted gamma lower bound respects location parameter."""
    shape, scale, location = 2.0, 3.0, 10.0
    BYTECODE = [
        (rv.OP_PUSH, shape),
        (rv.OP_PUSH, scale),
        (rv.OP_PUSH, location),
        (rv.OP_RAND_GAMMA, 0),
    ]
    opcodes  = np.array([op for op, _ in BYTECODE], dtype=np.double)
    operands = np.array([oe for _, oe in BYTECODE], dtype=np.double)

    vm = rv.VirtualMachine(opcodes, operands)
    d = vm.compute(samples=20_000)

    assert d.lower() >= location - 0.01  # allow tiny float tolerance


# --[ Node / high-level API tests ]---------------------------------------------

def test_gamma_node_mean():
    """Gamma node mean is approximately shape * scale."""
    shape, scale = 3.0, 4.0
    expected_mean = shape * scale  # 12.0

    d = _compute(rv.Gamma(shape=shape, scale=scale))
    assert abs(d.mean() - expected_mean) < 1.5


def test_gamma_node_location_shifts_min():
    """Location parameter shifts the distribution so lower bound >= location."""
    shape, scale, location = 2.0, 1.0, 7.0

    d = _compute(rv.Gamma(shape=shape, scale=scale, location=location))
    assert d.lower() >= location - 0.01


def test_gamma_node_location_shifts_mean():
    """Mean of shifted Gamma equals location + shape * scale."""
    shape, scale, location = 2.0, 3.0, 5.0
    expected_mean = location + shape * scale  # 11.0

    d = _compute(rv.Gamma(shape=shape, scale=scale, location=location))
    assert abs(d.mean() - expected_mean) < 1.5


def test_gamma_node_zero_location_default():
    """Default location of 0 gives same result as explicit location=0."""
    shape, scale = 4.0, 2.0

    d_default  = _compute(rv.Gamma(shape=shape, scale=scale))
    d_explicit = _compute(rv.Gamma(shape=shape, scale=scale, location=0))

    # Medians should be close (both are unshifted Gamma(4,2))
    assert abs(d_default.quantile(0.5) - d_explicit.quantile(0.5)) < 0.5


def test_gamma_node_shape_lt_one():
    """Gamma node works for shape < 1 (uses Ahrens-Dieter transform internally)."""
    shape, scale = 0.5, 2.0
    d = _compute(rv.Gamma(shape=shape, scale=scale))
    # mean ≈ 1.0; all samples should be positive
    assert d.lower() >= 0
    assert abs(d.mean() - shape * scale) < 1.0


def test_gamma_node_in_expression():
    """Gamma node can participate in arithmetic expressions."""
    shape, scale, location = 2.0, 2.0, 3.0
    # Expected mean of (Gamma + Constant(10)) = location + shape*scale + 10
    expr = rv.Gamma(shape=shape, scale=scale, location=location) + rv.Constant(10)
    d = _compute(expr)
    expected = location + shape * scale + 10  # 3+4+10 = 17
    assert abs(d.mean() - expected) < 1.5


def test_gamma_compile_produces_correct_opcodes():
    """Gamma node compiles to the expected bytecode."""
    node = rv.Gamma(shape=2.0, scale=3.0, location=5.0)
    codes, operands = node.compile()

    assert rv.OP_RAND_GAMMA in list(codes)
    # shape, scale, location should appear as PUSH operands before the gamma op
    gamma_idx = list(codes).index(rv.OP_RAND_GAMMA)
    push_operands = [operands[i] for i in range(gamma_idx) if codes[i] == rv.OP_PUSH]
    assert 2.0 in push_operands
    assert 3.0 in push_operands
    assert 5.0 in push_operands
