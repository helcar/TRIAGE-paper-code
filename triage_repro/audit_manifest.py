"""Validate that an FF++ c23 manifest follows the paper's split protocol."""
import argparse
import json
from .manifest import load_manifest, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(load_manifest(args.manifest)), indent=2))


if __name__ == "__main__":
    main()
