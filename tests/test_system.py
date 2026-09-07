import numpy as np
import pytest

from grape_scq import Transmon


def test_qubit_drift_hamiltonian_on_resonance():
    dt = 0.1e-9
    ref_amp = 10e6
    system = Transmon(n_levels=2, frequency=5e9, carrier_frequency=5e9, anharmonicity=-100e6, amplitude_scale=1.0)
    H0dt, HDdt, itH = system._hamiltonians(np.array([dt]), n_steps=1, reference_amplitude=ref_amp)
    assert np.allclose(H0dt[:, :, 0, 0], 0.0)

    expected_HD = np.zeros((2, 2))
    expected_HD[1, 0] = 0.5 * 2*np.pi*ref_amp*dt
    assert np.allclose(HDdt[:, :, 0, 0], expected_HD)


def test_qutrit_anharmonicity_only():
    dt = 0.1e-9
    alpha = -100e6
    system = Transmon(n_levels=3, frequency=5e9, carrier_frequency=5e9, anharmonicity=alpha)
    H0dt, HDdt, itH = system._hamiltonians(np.array([dt]), n_steps=1, reference_amplitude=10e6)
    expected_H0 = np.diag([0.0, 0.0, 2*np.pi*alpha]) * dt
    assert np.allclose(H0dt[:, :, 0, 0], expected_H0)


def test_ensemble_size_and_carrier_default():
    system = Transmon(frequency=[4.9e9, 5.0e9, 5.1e9])
    assert system.n_ensemble == 3
    assert system.carrier_frequency == pytest.approx(5.0e9)


def test_inconsistent_ensemble_sizes_raise():
    with pytest.raises(ValueError):
        Transmon(frequency=[4.9e9, 5.0e9, 5.1e9], amplitude_scale=[1.0, 1.1])


def test_sweep_builds_2d_grid_with_metadata():
    freqs = np.linspace(-1e6, 1e6, 3)
    amps = np.linspace(-0.1, 0.1, 4)
    system = Transmon.sweep(frequency=5e9 + freqs, amplitude_scale=1 + amps)
    assert system.n_ensemble == 12
    assert system.grid_shape == (3, 4)
    assert set(system.grid_axes) == {"frequency", "amplitude_scale"}
    assert np.allclose(system.grid_axes["frequency"], 5e9 + freqs)


def test_with_replaces_fields():
    system = Transmon(frequency=5e9)
    system2 = system.with_(frequency=5.1e9)
    assert system2.frequency[0, 0] == pytest.approx(5.1e9)
    assert system.frequency[0, 0] == pytest.approx(5e9)
