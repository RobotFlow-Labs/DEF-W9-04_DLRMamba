from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anima_dlrmamba.train import train_loop

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    train_loop(config_path=args.config, max_steps=args.max_steps)
