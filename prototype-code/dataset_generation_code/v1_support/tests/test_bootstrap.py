"""Regression test for the §1 project finder (PLAN v1.5).

The notebook must locate dataset_generation_code/ without trusting the
kernel cwd (server launched elsewhere, VS Code, nbconvert), and fail
loudly with the fix when v1_support/ is genuinely absent. Executes the
REAL §1 cell source from the built notebook — no duplicated logic.
Run:  python3 -m unittest v1_support.tests.test_bootstrap -v
"""
import json
import os
import subprocess as _sp
import sys
import tempfile
import types
import unittest
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_DIR))

# Mirror of §1's NEED_RENDER/PIP (the pip-spec contract — if §1's pins
# change, the spec assertion below fails until this is updated too).
RENDER_MODS = ["wfdb", "yaml", "PIL", "numpy", "scipy", "matplotlib",
               "pandas", "qrcode"]
RENDER_PINS = {"wfdb": "wfdb>=4.3", "yaml": "pyyaml", "PIL": "Pillow>=10",
               "numpy": "numpy>=1.24", "scipy": "scipy>=1.10",
               "matplotlib": "matplotlib>=3.7", "pandas": "pandas>=2",
               "qrcode": "qrcode"}


def _unimportable(importlib, mod: str) -> bool:
    # Mirrors the §1 check: missing OR present-but-broken (a version
    # clash raises at import with a non-ImportError type).
    try:
        importlib.import_module(mod)
        return False
    except Exception:
        return True


def section1_source() -> str:
    nb = json.loads((NB_DIR / "v1_image_generation.ipynb").read_text())
    src = nb["cells"][2]["source"]
    assert "_find_nb_dir" in src, "§1 cell moved; update this test"
    return src


class _FakeSubprocess:
    """Stands in for subprocess during the §1 exec: no real pip.

    Records pip invocations and simulates a successful install by
    injecting stub modules, so the cell's missing-detection + spec
    wiring is tested end to end without touching the environment.
    Fresh-interpreter probes (`python -c`) answer from probe_results.
    """

    def __init__(self, probe_results=None):
        self.calls: list = []
        self.probe_results = list(probe_results or [])

    def run(self, cmd, **kw):
        if cmd[1] == "-c":
            self.calls.append(("probe", list(cmd)))
            rc = self.probe_results.pop(0) if self.probe_results else 0
            out = _sp.CompletedProcess(cmd, rc)
            out.stderr = "" if rc == 0 else "simulated probe failure"
            return out
        assert cmd[1:3] == ["-m", "pip"], cmd
        self.calls.append(("pip", list(cmd)))
        for m in RENDER_MODS:
            sys.modules.setdefault(m, types.ModuleType(m))
        return _sp.CompletedProcess(cmd, 0)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)

    def _exec_cell(self, cwd, probe_results=None):
        import importlib
        expected = [RENDER_PINS[m] for m in RENDER_MODS
                    if _unimportable(importlib, m)]
        fake = _FakeSubprocess(probe_results)
        self._last_fake = fake  # inspectable when the cell raises
        real_sub = sys.modules.get("subprocess")
        pre = set(sys.modules)
        sys.modules["subprocess"] = fake
        try:
            os.chdir(cwd)
            ns: dict = {"__name__": "__main__"}
            exec(compile(section1_source(), "<nb-cell-2>", "exec"), ns)
        finally:
            if real_sub is not None:
                sys.modules["subprocess"] = real_sub
            else:
                del sys.modules["subprocess"]
            for k in set(sys.modules) - pre:
                del sys.modules[k]
        # the cell must pip-request exactly the missing specs (or nothing)
        # and probe the fresh interpreter once per install round
        pip_cmds = [c for k, c in fake.calls if k == "pip"]
        probes = [c for k, c in fake.calls if k == "probe"]
        self.assertLessEqual(len(pip_cmds), 2)
        if expected:
            self.assertGreaterEqual(len(pip_cmds), 1)
            tail = pip_cmds[0][pip_cmds[0].index("--timeout") + 2:]
            got = [a for a in tail if not a.startswith("-")]
            self.assertEqual(got, expected)
            self.assertGreaterEqual(len(probes), 1)
        else:
            self.assertEqual(fake.calls, [])
        return ns, fake

    def test_finds_project_from_parent(self):
        ns, _ = self._exec_cell(NB_DIR.parent)
        self.assertEqual(ns["NB_DIR"], NB_DIR)

    def test_finds_project_when_nested(self):
        (NB_DIR / "outputs").mkdir(exist_ok=True)
        ns, _ = self._exec_cell(NB_DIR / "outputs")
        self.assertEqual(ns["NB_DIR"], NB_DIR)

    def _shadow_broken(self, tmp, mod="qrcode"):
        Path(tmp, f"{mod}.py").write_text(
            "raise TypeError('simulated broken install')\n")
        sys.path.insert(0, tmp)
        return sys.modules.pop(mod, None)

    def _unshadow(self, tmp, mod, saved):
        sys.path.remove(tmp)
        if saved is not None:
            sys.modules[mod] = saved
        else:
            sys.modules.pop(mod, None)

    def _pip_calls(self):
        return [c for k, c in self._last_fake.calls if k == "pip"]

    def _probe_calls(self):
        return [c for k, c in self._last_fake.calls if k == "probe"]

    def test_broken_present_package_demands_restart(self):
        # Present-but-broken (raises TypeError at import, like the
        # wfdb 4.1.2 / pandas 3 clash): pip-requested, fresh-probed,
        # then a restart demand — never a same-process re-import.
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._shadow_broken(tmp)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    self._exec_cell(NB_DIR.parent)
            finally:
                self._unshadow(tmp, "qrcode", saved)
        self.assertIn("RESTART the kernel", str(ctx.exception))
        pip = self._pip_calls()
        self.assertEqual(len(pip), 1)
        self.assertIn("qrcode", " ".join(pip[0]))
        self.assertEqual(len(self._probe_calls()), 1)

    def test_probe_failure_force_reinstalls_then_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._shadow_broken(tmp)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    # probe fails once (force-reinstall path), then passes;
                    # still a restart demand because qrcode was 'broken'
                    self._exec_cell(NB_DIR.parent, probe_results=[1, 0])
            finally:
                self._unshadow(tmp, "qrcode", saved)
        self.assertIn("RESTART the kernel", str(ctx.exception))
        pip = self._pip_calls()
        self.assertEqual(len(pip), 2)
        self.assertIn("--force-reinstall", pip[1])
        self.assertIn("--no-cache-dir", pip[1])
        self.assertEqual(len(self._probe_calls()), 2)

    def test_probe_failure_twice_raises_manual_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._shadow_broken(tmp)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    self._exec_cell(NB_DIR.parent, probe_results=[1, 1])
            finally:
                self._unshadow(tmp, "qrcode", saved)
        self.assertIn("Manual fix", str(ctx.exception))
        self.assertIn(sys.executable, str(ctx.exception))
        self.assertEqual(len(self._pip_calls()), 2)
        self.assertEqual(len(self._probe_calls()), 2)

    def test_missing_project_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                exec(compile(section1_source(), "<nb-cell-2>", "exec"),
                     {"__name__": "__main__"})
        msg = str(ctx.exception)
        self.assertIn("WHOLE dataset_generation_code", msg)
        self.assertIn("%cd", msg)


if __name__ == "__main__":
    unittest.main()
