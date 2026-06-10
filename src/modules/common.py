from pathlib import Path

dir_project = Path(__file__).parent.parent.parent
dir_tmp = dir_project / "tmp"
dir_tmp.mkdir(exist_ok=True, parents=True)
