import numpy as np

from retrieval_cerebellum.sqp_skill_adapter import candidate_sample_positions


def test_full_suffix_sampling_reaches_demonstration_end():
    np.testing.assert_allclose(
        candidate_sample_positions(11, 2),
        [0.0, 5.0, 10.0],
    )


def test_local_sampling_does_not_compress_full_suffix():
    np.testing.assert_allclose(
        candidate_sample_positions(11, 2, source_span_steps=2),
        [0.0, 1.0, 2.0],
    )
