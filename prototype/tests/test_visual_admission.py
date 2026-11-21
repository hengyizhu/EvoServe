import json, os, tempfile, unittest
from pathlib import Path
from collections import deque
from unittest.mock import Mock, patch
from evoserve.branches import Branches


class VisualPlacement(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.d = Path(self.scratch.name)
        self.b = Branches.__new__(Branches)
        self.b.out = self.d
        self.b.waiting = deque([0])
        self.b.next_admission = 0
        self.b.preparing = None
        self.b.job = None
        self.b.vision = {}
        self.b.rebuild_refs = {}
        self.b.retired_visual = set()
        self.b.train = {}
        self.pool = {0: [{} for _ in range(32)]}

    def test_low_memory_defers_and_keeps_queue(self):
        p = self.d / "peak.json"
        p.write_text('{"peak_mib":1000}')
        with (
            patch.dict(
                os.environ, EVOSERVE_VISUAL_CARD="2", EVOSERVE_VISUAL_PEAK=str(p)
            ),
            patch(
                "evoserve.gpu_inventory.inventory",
                return_value={2: dict(free_mib=1099, processes=[])},
            ),
        ):
            self.b.service_visual(self.pool)
            self.b.service_visual(self.pool)
        lines = (self.d / "placement.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["reason"], "insufficient_free_memory")
        self.assertEqual(list(self.b.waiting), [0])

    def test_child_failure_does_not_end_loop(self):
        proc = Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        self.b.job = (0, proc, self.d)
        self.b.waiting.clear()
        self.b.service_visual(self.pool)
        self.b.service_visual(self.pool)
        self.assertIsNone(self.b.job)
        self.assertIn(0, self.b.retired_visual)

    def test_corrupt_transfer_does_not_end_loop(self):
        proc = Mock()
        proc.poll.return_value = 0
        proc.returncode = 0
        self.b.job = (0, proc, self.d)
        self.b.waiting.clear()
        with (
            patch("torch.cuda.device"),
            patch("torch.load", side_effect=OSError("damaged payload")),
        ):
            self.b.service_visual(self.pool)
        self.assertEqual(
            json.loads((self.d / "placement.jsonl").read_text())["event"],
            "visual_failed",
        )
