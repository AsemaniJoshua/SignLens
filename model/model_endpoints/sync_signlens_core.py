"""Copy notebooks/signlens_core into this server folder.

The Docker image is built from model/model_endpoints only, so the server keeps
its own copy of the shared code. Run this after changing notebooks/signlens_core:

    python model/model_endpoints/sync_signlens_core.py

tests/test_service.py fails if the two copies differ.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "notebooks", "signlens_core"))
DST = os.path.join(HERE, "signlens_core")

if __name__ == "__main__":
    shutil.rmtree(DST, ignore_errors=True)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"copied {SRC} -> {DST}")
