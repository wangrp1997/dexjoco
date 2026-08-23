import numpy as np

from retrieval_cerebellum.contact_response_search import (
    BoundedContactResponseSearch,
    ContactSearchObservation,
    ContactSearchStrategy,
)


def _observation(force_right=0.0, force_left=0.0, stability=(1.0, 1.0)):
    wrench = np.zeros((2, 6))
    wrench[0, 0] = force_right
    wrench[1, 0] = force_left
    return ContactSearchObservation(wrench, np.asarray(stability))


def test_search_descends_before_contact():
    search = BoundedContactResponseSearch([0.0, 0.0, 1.0], "fixed_spiral")

    command = search.step(_observation())

    assert command.phase == "descent"
    assert command.relative_translation_world[2] > 0.0


def test_adaptive_search_runs_symmetric_probe_sequence():
    search = BoundedContactResponseSearch([0.0, 0.0, 1.0], "adaptive_single")

    phases = [search.step(_observation(force_right=4.0)).phase for _ in range(7)]

    assert phases[:3] == ["probe_0_1", "probe_0_-1", "probe_0_0"]
    assert phases[-1] == "adaptive_correction"


def test_bimanual_fraction_avoids_loaded_wrist():
    search = BoundedContactResponseSearch(
        [0.0, 0.0, 1.0], ContactSearchStrategy.ADAPTIVE_BIMANUAL
    )

    command = search.step(_observation(force_right=8.0, force_left=1.0))

    assert command.right_motion_fraction < 0.5


def test_hard_force_retreats():
    search = BoundedContactResponseSearch([0.0, 0.0, 1.0], "adaptive_bimanual")

    command = search.step(_observation(force_right=80.0))

    assert command.retreat
    assert command.phase == "hard_force_retreat"


def test_contact_search_releases_after_force_clears():
    search = BoundedContactResponseSearch([0.0, 0.0, 1.0], "adaptive_bimanual")
    assert search.step(_observation(force_right=4.0)).phase.startswith("probe_")

    phases = [search.step(_observation()).phase for _ in range(3)]

    assert phases[-1] == "descent"
