"""Audit source-group separation for Celeb-DF-v2 and DFDC manifests.

Source-group-disjoint is NOT equivalent to person-identity-disjoint for Celeb-DF.
The DFDC test split in the paper is a user-created held-out training subset.
"""
import argparse
import json

from .evaluate_final import read_split_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    by_split = read_split_manifest(a.manifest)
    result = {
        "protocol": "source-video-group disjoint, not necessarily identity disjoint",
        "partition_frames": {key: len(rows) for key, rows in by_split.items()},
        "partition_groups": {key: len({r["group_id"] for r in rows})
                             for key, rows in by_split.items()},
        "partition_real_fake": {
            key: {"real": sum(r["label"] == "0" for r in rows),
                  "fake": sum(r["label"] == "1" for r in rows)}
            for key, rows in by_split.items()
        },
        "note": "Preserve official Celeb-DF test list; DFDC source groups derive from original real videos.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
