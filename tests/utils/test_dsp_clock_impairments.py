"""The vectorized sampling_clock_impairments must reproduce the original per-sample loop exactly:
same output values and length, and the random generator left in the same state."""

import numpy as np
import pytest

from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    TorchSigRealDataType,
    partition_polyphase,
    prototype_polyphase_filter,
    sampling_clock_impairments,
)


def _reference_partition_polyphase(h, up_rate, taps_per_phase):
    h_pfb = np.zeros((up_rate, taps_per_phase), dtype=TorchSigRealDataType)
    for phase in range(up_rate):
        tap_idx = phase
        for idx in range(taps_per_phase):
            h_pfb[phase, idx] = h[tap_idx] * up_rate if tap_idx < len(h) else 0.0
            tap_idx += up_rate
    return h_pfb


def _reference_sampling_clock_impairments(h, x, uprate, drate, jitter_ppm, drift_ppm, rng=None):
    """The original per-sample implementation (TorchSig 2.1.1), kept as the specification."""
    taps_per_phase = int(np.ceil(len(h) / uprate))
    h_pfb = _reference_partition_polyphase(h, uprate, taps_per_phase)
    padded_len = len(x) + 2 * taps_per_phase - 1
    input_padded = np.zeros(padded_len, dtype=TorchSigComplexDataType)
    start = taps_per_phase - 1
    input_padded[start:start + len(x)] = x
    q_step = uprate / drate
    num_output_samples = int(np.ceil(len(input_padded) * uprate / drate)) + 1
    output_samples = np.zeros(num_output_samples, dtype=TorchSigComplexDataType)
    input_idx = output_idx = 0
    clock_drift = 0.0
    idx_stop = len(input_padded) - taps_per_phase
    jitter_std, drift_std = jitter_ppm * 1e-6, drift_ppm * 1e-6
    while input_idx < idx_stop:
        while q_step >= uprate:
            q_step -= uprate
            input_idx += 1
        delay_slice = input_padded[input_idx:input_idx + taps_per_phase]
        if q_step >= uprate:
            break
        h_phase = h_pfb[int(q_step)][:taps_per_phase]
        acc_re = np.sum(h_phase * delay_slice[::-1].real)
        acc_im = np.sum(h_phase * delay_slice[::-1].imag)
        output_samples[output_idx] = acc_re + 1j * acc_im
        output_idx += 1
        if jitter_ppm != 0.0 or drift_ppm != 0.0:
            clock_jitter = rng.normal(0.0, jitter_std)
            clock_drift += rng.normal(0.0, drift_std)
            q_step += drate + clock_jitter + clock_drift
        else:
            q_step += drate
    return output_samples[:output_idx - 1] if output_idx > 0 else np.array([], dtype=TorchSigComplexDataType)


@pytest.fixture(scope="module")
def prototype():
    return prototype_polyphase_filter(num_branches=5000).astype(TorchSigRealDataType)


def test_partition_polyphase_matches_reference():
    h = np.random.default_rng(0).standard_normal(1003).astype(TorchSigRealDataType)
    for up, taps in ((7, 144), (10, 101), (5000, 1)):
        assert np.array_equal(partition_polyphase(h, up, taps), _reference_partition_polyphase(h, up, taps))


@pytest.mark.parametrize("jitter_ppm, drift_ppm", [(10, 0), (0, 10), (37.5, 0), (0, 3.2), (5, 5), (0, 0)])
@pytest.mark.parametrize("length", [1, 64, 1000, 4096])
def test_sampling_clock_impairments_matches_reference(prototype, jitter_ppm, drift_ppm, length):
    for seed in range(3):
        g = np.random.default_rng(100 * seed + length)
        x = (g.standard_normal(length) + 1j * g.standard_normal(length)).astype(TorchSigComplexDataType)
        r_ref, r_new = np.random.default_rng(seed), np.random.default_rng(seed)
        expected = _reference_sampling_clock_impairments(prototype, x, 5000, 5000, jitter_ppm, drift_ppm, r_ref)
        actual = sampling_clock_impairments(prototype, x, 5000, 5000, jitter_ppm, drift_ppm, r_new)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
        assert r_new.bit_generator.state == r_ref.bit_generator.state  # downstream randomness unchanged


def test_sampling_clock_impairments_other_rates():
    """Non-default rates (as used elsewhere in DSP code paths) also match."""
    h = prototype_polyphase_filter(num_branches=32).astype(TorchSigRealDataType)
    x = (np.random.default_rng(3).standard_normal(500) + 1j).astype(TorchSigComplexDataType)
    for drate in (32, 31.5, 40):
        r_ref, r_new = np.random.default_rng(9), np.random.default_rng(9)
        np.testing.assert_array_equal(
            sampling_clock_impairments(h, x, 32, drate, 20, 2, r_new),
            _reference_sampling_clock_impairments(h, x, 32, drate, 20, 2, r_ref),
        )
        assert r_new.bit_generator.state == r_ref.bit_generator.state
