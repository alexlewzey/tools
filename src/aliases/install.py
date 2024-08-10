"""Add command to source alias.sh file to the .(bash|zsh)rc file."""
from pathlib import Path

bashrc_path = Path.home()/'.bashrc'
zshrc_path = Path.home()/'.zshrc'
rc_path: Path
if zshrc_path.exists():
    rc_path = zshrc_path
elif bashrc_path.exists():
    rc_path = bashrc_path
else:
    raise Exception(f'Neither {bashrc_path}  or {zshrc_path} exist.')
aliases_path = Path(__file__).parent/'aliases.sh'
cmd: str = f'source "{aliases_path.as_posix()}"'
with rc_path.open('r') as f:
    content = f.read()
if cmd not in content:
    with rc_path.open('a') as f:
        f.write('\n\n'+ cmd+ '\n')