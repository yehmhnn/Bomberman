"""Regression test for the tournament's single-directory copy process."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parent


class IsolatedPPOPackagingTests(unittest.TestCase):
    def test_ppo_imports_without_repository_shared_or_dqn_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent_code = root / "agent_code"
            agent_code.mkdir()
            (agent_code / "__init__.py").touch()
            shutil.copytree(REPO_ROOT / "agent_code" / "ppo_agent", agent_code / "ppo_agent")
            for filename in ("settings.py", "events.py", "fallbacks.py"):
                shutil.copy2(REPO_ROOT / filename, root / filename)

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            environment["PYTHONPYCACHEPREFIX"] = str(root / "pycache")
            command = [
                sys.executable,
                "-c",
                (
                    "from types import SimpleNamespace; import numpy as np; "
                    "from agent_code.ppo_agent import callbacks, train; "
                    "logger=SimpleNamespace(info=lambda *a,**k:None,debug=lambda *a,**k:None); "
                    "learner=SimpleNamespace(train=False,logger=logger); callbacks.setup(learner); "
                    "field=-np.ones((7,7),dtype=int); field[1:6,1:6]=0; "
                    "state={'field':field,'self':('me',0,True,(3,3)),'others':[],"
                    "'bombs':[],'coins':[(4,3)],'explosion_map':np.zeros_like(field)}; "
                    "assert callbacks.act(learner,state) in callbacks.ACTIONS"
                ),
            ]
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
