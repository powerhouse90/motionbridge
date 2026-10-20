"""Machine-specific settings from config.toml at the repository root."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def load(path=None):
    path = Path(path) if path else ROOT / 'config.toml'
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found; copy config.example.toml to config.toml and edit it')
    config = tomllib.loads(path.read_text(encoding='utf-8'))
    paths = config.setdefault('paths', {})
    paths['runs'] = str((ROOT / paths.get('runs', 'runs')).resolve())
    config.setdefault('rigs', [])
    config.setdefault('motions', [])
    return config
