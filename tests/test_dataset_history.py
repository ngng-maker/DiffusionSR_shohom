"""History stacking for n_steps > 1 (SimulationXZDataset._history_index), without data."""
from diffusionsr.datasets.dataset import SimulationXZDataset


def fake(history, n_frames=(3, 4)):
    ds = object.__new__(SimulationXZDataset)
    ds.history = history
    ds.hr_paths = [f'/r/train/HR/power{100 + 30 * c}velocity500/_{1001 + f}.npy'
                   for c, n in enumerate(n_frames) for f in range(n)]
    return ds


def test_consecutive_same_simulation_then_ambient():
    ds = fake('consecutive')               # sim A: idx 0-2, sim B: idx 3-6
    n = 5
    slots = lambda i: [ds._history_index(i, step) for step in reversed(range(2, n + 1))]   # slots t-4..t-1
    assert slots(6) == [None, 3, 4, 5]     # t-1, t-2, t-3 from sim B; t-4 would be sim A -> ambient
    assert slots(3) == [None, None, None, None]   # first frame of sim B: no history from sim A
    assert slots(2) == [None, None, 0, 1]


def test_legacy_reproduces_original_stacking():
    ds = fake('legacy')
    assert [ds._history_index(6, s) for s in (3, 2)] == [3, 4]      # t-3, t-2 (t-1 skipped)
    assert [ds._history_index(3, s) for s in (3, 2)] == [None, 1]   # crosses into sim A
