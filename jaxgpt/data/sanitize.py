from pathlib import Path

def prepare_file(path: Path) -> list[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]