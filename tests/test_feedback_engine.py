"""Tests for the fatigue/posture decision engine and the coach: python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import polar_coach  # noqa: E402
from feedback_engine import (  # noqa: E402
    Decision,
    Gate,
    choose_primary,
    fatigue_cause,
    fatigue_level,
    angle_level,
)
from polar_simulator import simulated_events  # noqa: E402


class FatigueRuleTests(unittest.TestCase):
    def test_levels_need_two_of_three(self):
        self.assertEqual(fatigue_level(0, 0, 0), 0)
        self.assertEqual(fatigue_level(6, -16, 0), 1)       # HR + RMSSD
        self.assertEqual(fatigue_level(12, -26, -21), 2)
        self.assertEqual(fatigue_level(12, -36, -31), 3)    # RMSSD + SDNN are enough
        self.assertEqual(fatigue_level(30, -45, -10), 4)    # HR + RMSSD
        self.assertEqual(fatigue_level(30, 0, 0), 0)        # only one condition

    def test_cause_patterns(self):
        self.assertEqual(fatigue_cause(12, -30, -30), "cardiovascular")
        self.assertEqual(fatigue_cause(20, -30, -30), "prolonged_exertion")
        self.assertEqual(fatigue_cause(20, -20, -20), "heat")
        self.assertEqual(fatigue_cause(7, -45, -30), "sleep")
        self.assertIsNone(fatigue_cause(2, -5, -5))

    def test_long_work_breaks_the_tie_with_cardiovascular(self):
        self.assertEqual(fatigue_cause(12, -30, -30, minutes_on_shift=300), "long_work")

    def test_hot_skin_means_heat(self):
        self.assertEqual(fatigue_cause(12, -30, -30, skin_temp=36.0), "heat")


class PostureRuleTests(unittest.TestCase):
    def test_angle_levels(self):
        self.assertEqual(angle_level(10, 5), (1, None))
        self.assertEqual(angle_level(25, 0), (2, "forward_bending"))
        self.assertEqual(angle_level(50, 0), (3, "forward_bending"))
        self.assertEqual(angle_level(0, 12), (2, "side_bending"))
        self.assertEqual(angle_level(25, 25), (3, "side_bending"))  # worse cause wins


class GateTests(unittest.TestCase):
    def run_gate(self, gate, levels, step=5):
        fired = []
        for i, level in enumerate(levels):
            t = i * step
            gate.observe(t, level)
            ready = gate.ready(t, level)
            if ready:
                gate.mark_spoken(t, ready)
                fired.append((t, ready))
        return fired

    def test_level_must_hold_for_its_duration(self):
        gate = Gate({2: 30, 3: 60}, repeat_s=600)
        self.assertEqual(self.run_gate(gate, [2] * 5), [])           # 20 s only
        gate = Gate({2: 30, 3: 60}, repeat_s=600)
        self.assertEqual(self.run_gate(gate, [2] * 8), [(30, 2)])    # fires at 30 s, once

    def test_escalation_and_repeat(self):
        gate = Gate({2: 30, 3: 60}, repeat_s=60)
        fired = self.run_gate(gate, [2] * 8 + [3] * 20)
        self.assertEqual(fired[0], (30, 2))
        self.assertIn((40 + 60, 3), fired)          # level 3 after holding 60 s
        self.assertGreater(len(fired), 2)           # and repeats while it continues

    def test_rapid_jump_skips_the_wait(self):
        gate = Gate({2: 30, 3: 60, 4: 120}, repeat_s=600, rapid_jump=2, rapid_window_s=60)
        self.assertEqual(self.run_gate(gate, [1, 1, 4]), [(10, 4)])


class ArbiterTests(unittest.TestCase):
    def d(self, track, urgency):
        return Decision(track, 0, 2, "", None, "", "", "", "", urgency)

    def test_most_urgent_wins_and_posture_wins_ties(self):
        self.assertEqual(choose_primary([self.d("fatigue", 2), self.d("posture", 1)]).track, "fatigue")
        self.assertEqual(choose_primary([self.d("fatigue", 1), self.d("posture", 1)]).track, "posture")
        self.assertIsNone(choose_primary([None, None]))


class SimulatedShiftTests(unittest.TestCase):
    """End to end: the 50-minute simulated shift must produce the scripted feedback."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp()
        polar_coach.LOG_PATH = Path(tmp) / "log.csv"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            polar_coach.run(simulated_events(seed=7), speed=0, dry_run=True)
        cls.lines = [line for line in out.getvalue().splitlines() if "->" in line]
        cls.text = "\n".join(cls.lines)

    def test_posture_feedback(self):
        self.assertIn("POSTURE L2 Poor Posture / Forward Bending", self.text)
        self.assertIn("POSTURE L3 Critical Posture / Forward Bending", self.text)
        self.assertIn("POSTURE L2 Poor Posture / Side Bending", self.text)
        self.assertIn("POSTURE L3 Critical Posture / Static Posture", self.text)

    def test_fatigue_feedback_escalates(self):
        self.assertIn("FATIGUE L2 Medium Fatigue / Cardiovascular Fatigue", self.text)
        self.assertIn("FATIGUE L3 High Fatigue / Prolonged Physical Exertion", self.text)
        self.assertIn("FATIGUE L4 Very High Fatigue", self.text)

    def test_silent_during_baseline_and_recovery(self):
        minutes = [int(line.split()[1]) for line in self.lines]
        self.assertTrue(all(m >= 5 for m in minutes))                  # nothing during baseline
        self.assertFalse(any(32 <= m <= 39 for m in minutes))           # nothing while recovered

    def test_log_has_closed_loop_action(self):
        log = polar_coach.LOG_PATH.read_text()
        self.assertIn("closed_loop_action", log.splitlines()[0])
        self.assertIn("Reduce trunk flexion immediately", log)


class ReplayTests(unittest.TestCase):
    def test_replay_reads_a_recording_in_time_order(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "hr.csv").write_text("t,hr,rr_ms\n1.0,75,800\n2.0,75,800\n")
        (folder / "acc.csv").write_text("t,ax,ay,az\n0.5,1,0,0\n1.5,1,0,0\n")
        kinds = [ev[0] for ev in polar_coach.replay_events(folder)]
        self.assertEqual(kinds, ["acc", "beat", "acc", "beat"])


if __name__ == "__main__":
    unittest.main()
