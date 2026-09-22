"""Oracle trust gate primitives: _ulp_jitter and _reference_stable.

CPU-only (the harnesses' own GPU behavior is covered by campaigns and the
GPU smoke test). Locks in the semantics the gates rely on: exactly one ulp
of perturbation, zeros moved off zero, integer dtypes untouched, and the
fixed 1e-2 self-error threshold of _reference_stable.
"""
import unittest

import torch

from src.workflow.emitter.region_checks import _ulp_jitter, _reference_stable


class OracleGateTests(unittest.TestCase):
    def test_ulp_jitter_fp16_semantics(self):
        x = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float16)
        j = _ulp_jitter(x)
        # nextafter(y, sign(y)): positives move away from zero, negatives
        # move toward zero, by exactly one ulp; zeros move off zero first.
        self.assertEqual(j.dtype, torch.float16)
        self.assertEqual(j.device, x.device)
        self.assertEqual(j[0].item(), torch.nextafter(
            torch.tensor(torch.finfo(torch.float16).tiny, dtype=torch.float16),
            torch.tensor(1.0, dtype=torch.float16)).item())
        self.assertEqual(j[1].item(), torch.nextafter(x[1], x[1].sign()).item())
        self.assertEqual(j[2].item(), torch.nextafter(x[2], x[2].sign()).item())
        self.assertNotEqual(j[0].item(), 0.0)
        # The input tensor is never mutated.
        self.assertEqual(x.tolist(), [0.0, 1.0, -2.0])

    def test_ulp_jitter_integer_types_are_noops(self):
        for dtype in (torch.int8, torch.int32):
            x = torch.tensor([1, -2, 3], dtype=dtype)
            j = _ulp_jitter(x)
            self.assertTrue(torch.equal(j, x))
            self.assertEqual(j.dtype, dtype)

    def test_reference_stable_accepts_benign_and_rejects_gross_disagreement(self):
        ref = torch.randn(8, 8)
        # fp64 copy that matches: stable.
        self.assertTrue(_reference_stable(ref, ref.to(torch.float64)))
        # fp64 copy that disagrees beyond 1e-2 self-relative: unstable.
        self.assertFalse(_reference_stable(ref, ref.to(torch.float64) * 100.0))
        # Jittered copy of benign values stays stable (fp16 noise is ~5e-4
        # relative; benign programs amplify at most ~2x).
        half = (torch.randn(8, 8) * 0.1).to(torch.float16)
        self.assertTrue(_reference_stable(
            half.to(torch.float32), half.to(torch.float64),
            _ulp_jitter(half).to(torch.float32)))


if __name__ == '__main__':
    unittest.main()
