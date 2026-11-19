"""Download the authors' IPN HandS annotations/MediaPipe Hands data, SHA-256 pinned.

Source: https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md
Data license: CC BY 4.0. See docs/LEARNED_SWIPES.md for attribution.
No webcam recording or upload is performed.
"""

import argparse
import hashlib
from pathlib import Path
import urllib.request


FILES = {
    "annotations": ("IPN_HandS_annotations.zip", "1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1",
                    "520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec"),
    "hands": ("IPN_HandS_landmarks_MPHands.zip", "1Y-3_MUHJ6OYpv-BNof5vYDRR4YNgrGQ2",
              "e511bb2b902207b82a296190daadc30a057f3abc16f628579542d2b1274ce598"),
}
# The annotation pin is the observed official HTTPS download on 2026-10-20.
# It differs from the author's README pin (4686a2...ec71); see provenance notes.
# The landmark archive matches the publisher's checksum exactly.


def fetch(kind, directory):
    name, file_id, expected = FILES[kind]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Existing file has a different checksum: {target}; move it aside first")
        print(f"Already verified: {target}", flush=True)
        return target
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    temporary = target.with_suffix(".zip.part")
    digest = hashlib.sha256()
    count = 0
    print(f"Downloading {name} from the authors' public Google Drive...", flush=True)
    # A partial file can be retried; a complete existing archive is never overwritten.
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Google Drive returned a web page instead of an archive; use the author's download link")
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
            count += len(block)
            if count % (50 * 1024 * 1024) == 0:
                print(f"  {count // (1024 * 1024)} MiB", flush=True)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {temporary}; archive was not accepted")
    temporary.rename(target)
    print(f"Verified {count:,} bytes: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=FILES)
    parser.add_argument("--output", type=Path, default=Path("datasets/ipn_hands"))
    args = parser.parse_args()
    for kind in ([args.only] if args.only else FILES):
        fetch(kind, args.output)


if __name__ == "__main__":
    main()
