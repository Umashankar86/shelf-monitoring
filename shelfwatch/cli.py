import argparse
import json
import logging

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Proglint Shelfwatch local application")
    parser.add_argument("command", choices=["serve", "doctor", "index"], nargs="?", default="serve")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
    elif args.command == "doctor":
        from .models import Detector, Recognizer, bundle_status

        print(f"Colab exports: {cfg['artifacts']}")
        for item in bundle_status(cfg):
            print(f"{'FOUND' if item['present'] else 'MISSING':7} {item['file']}")
        try:
            Detector(cfg)
            Recognizer(cfg)
        except Exception as exc:
            logging.getLogger(__name__).debug("Bundle validation failed", exc_info=True)
            print(f"Real monitoring not ready: {exc}")
            raise SystemExit(1)
        print("Real monitoring bundle loads successfully.")
    else:
        from .models import build_index

        print(json.dumps(build_index(cfg), indent=2))


if __name__ == "__main__":
    main()
