"""Regression test for the tournament's single-agent packaging process."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent
AGENTS = ("tabular_q_agent", "tabular_sarsa_agent")


class IsolatedTabularPackagingTests(unittest.TestCase):
    def test_each_agent_imports_without_repository_shared_package(self):
        for agent in AGENTS:
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as directory:
                isolated_root = Path(directory)
                isolated_agent_code = isolated_root / "agent_code"
                isolated_agent_code.mkdir()
                (isolated_agent_code / "__init__.py").touch()

                shutil.copytree(
                    REPO_ROOT / "agent_code" / agent,
                    isolated_agent_code / agent,
                )
                shutil.copy2(REPO_ROOT / "settings.py", isolated_root / "settings.py")
                shutil.copy2(REPO_ROOT / "events.py", isolated_root / "events.py")
                shutil.copy2(REPO_ROOT / "fallbacks.py", isolated_root / "fallbacks.py")

                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(isolated_root)
                environment["PYTHONPYCACHEPREFIX"] = str(isolated_root / "pycache")
                command = [
                    sys.executable,
                    "-c",
                    (
                        "from types import SimpleNamespace; import numpy as np; "
                        f"from agent_code.{agent} import callbacks, train; "
                        "logger=SimpleNamespace(info=lambda *a,**k:None, "
                        "debug=lambda *a,**k:None); "
                        "learner=SimpleNamespace(train=False,logger=logger); "
                        "callbacks.setup(learner); "
                        "field=-np.ones((7,7),dtype=int); field[1:6,1:6]=0; "
                        "state={'field':field,'self':('me',0,True,(3,3)),"
                        "'others':[],'bombs':[],'coins':[(4,3)],"
                        "'explosion_map':np.zeros_like(field)}; "
                        "assert callbacks.act(learner,state) in callbacks.ACTIONS"
                    ),
                ]
                result = subprocess.run(
                    command,
                    cwd=isolated_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
