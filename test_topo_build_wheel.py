import os
from pathlib import Path
import subprocess
import sys


def test_built_wheel_imports_outside_repository(tmp_path):
    repository = Path(__file__).resolve().parent
    dist = tmp_path / "dist"
    target = tmp_path / "site"
    dist.mkdir()
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(dist), str(repository)],
        check=True,
        cwd=tmp_path,
    )
    wheel = next(dist.glob("topo_build-*.whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target",
         str(target), str(wheel)],
        check=True,
        cwd=tmp_path,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    subprocess.run(
        [sys.executable, "-c",
         "from topo_build import build_topology, write_artifacts"],
        check=True,
        cwd=tmp_path,
        env=environment,
    )
