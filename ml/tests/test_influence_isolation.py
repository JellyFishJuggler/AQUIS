"""Isolation guards for the research GNN influence feature.

These tests assert the *hard* constraint: nothing the influence feature added can
change how the existing app behaves. Concretely:

* the JSON loader imports no torch / torch_geometric (so the API stays light);
* importing ``api`` never pulls in the research package or PyG;
* no serving module references the research package;
* the training CLI refuses to write outside ``models/research/``;
* serving the influence endpoints leaves ``runtime.json`` byte-identical.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tests._helpers import has

ML = Path(__file__).resolve().parents[1]          # ml/


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=str(ML),
                          capture_output=True, text=True, timeout=300)


@unittest.skipUnless(has("api.py"), "api module missing")
class TestImportIsolation(unittest.TestCase):
    def test_loader_imports_no_torch(self):
        r = _run("import sys; import serving.influence as I; "
                 "print(sorted(m for m in ('torch','torch_geometric') "
                 "if m in sys.modules))")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(r.stdout.strip(), "[]", r.stdout)

    def test_api_import_excludes_research_and_pyg(self):
        r = _run("import sys; import api; "
                 "print(sorted(m for m in ('research','torch_geometric') "
                 "if m in sys.modules))")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(r.stdout.strip(), "[]", r.stdout)

    def test_serving_modules_do_not_reference_research(self):
        import re
        offenders = []
        pattern = re.compile(r"^\s*(?:from|import)\s+(?:research|torch_geometric)\b",
                             re.MULTILINE)
        for p in (ML / "serving").glob("*.py"):
            if pattern.search(p.read_text(encoding="utf-8")):
                offenders.append(p.name)
        self.assertEqual(offenders, [])

    def test_api_module_does_not_import_research(self):
        text = (ML / "api.py").read_text(encoding="utf-8")
        self.assertNotIn("import research", text)
        self.assertNotIn("research.gnn", text)
        self.assertNotIn("torch_geometric", text)


class TestTrainingWriteGuard(unittest.TestCase):
    def test_refuses_to_write_outside_models_research(self):
        target = ML / "models" / "_guard_should_not_exist.pt"
        if target.exists():                            # pragma: no cover
            target.unlink()
        r = _run("from research.gnn.train import main; import sys; "
                 "sys.argv = ['x', '--out', 'models/_guard_should_not_exist.pt']; "
                 "main()")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to write outside models/research/",
                      r.stdout + r.stderr)
        self.assertFalse(target.exists())

    def test_train_only_targets_models_research(self):
        text = (ML / "research" / "gnn" / "train.py").read_text(encoding="utf-8")
        self.assertIn('if out.parent.name != "research"', text)
        # the production checkpoints are never opened for writing
        for prod in ("traj_dl.pt", "pooled_dl.pt"):
            self.assertNotIn(f'"{prod}"', text)


@unittest.skipUnless(has("api.py"), "api module missing")
class TestRuntimeUntouched(unittest.TestCase):
    def test_runtime_json_is_not_modified(self):
        runtime = ML / "data" / "refresh" / "runtime.json"
        before = runtime.read_bytes() if runtime.exists() else None
        import api
        client = api.app.test_client()
        for url in ("/influence/meta", "/influence/graph",
                    "/influence/station/x", "/influence/district/x"):
            client.get(url)
        after = runtime.read_bytes() if runtime.exists() else None
        self.assertEqual(before, after)

    def test_research_artifacts_live_under_outputs(self):
        from research.gnn import influence as R
        self.assertEqual(R.OUT.parent.name, "outputs")
        self.assertEqual(R.RESEARCH.name, "research")
        self.assertEqual(R.RESEARCH.parent.name, "models")


if __name__ == "__main__":
    unittest.main()
