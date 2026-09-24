"""Smoke-test the hosted defensive coverage Workflow on a UGA All-22 still.

The saved Workflow is called by workspace + workflow ID. Weights and the graph
stay on Roboflow. The API key comes from ROBOFLOW_API_KEY in .env — never commit it.

This Workflow's field calibration is fixed to the UGA camera. Coverage still
comes from the Qwen endpoint, not the newly trained ResNet18.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# 1. Import the library
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

IMAGE = ROOT / "uga-play.jpg"
OUT = ROOT / "coverage_test_out"


def _write_image(value, path: Path) -> bool:
    blob = value
    if isinstance(value, dict):
        blob = value.get("value") or value.get("data")
    if isinstance(value, (bytes, bytearray)):
        path.write_bytes(bytes(value))
        return True
    if not isinstance(blob, str) or not blob:
        return False
    if "," in blob and blob.strip().startswith("data:"):
        blob = blob.split(",", 1)[1]
    path.write_bytes(base64.b64decode(blob))
    return True


OUT.mkdir(exist_ok=True)

api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not api_key:
    raise SystemExit(
        "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add the key "
        "from Workspace Settings → API Keys."
    )
if not IMAGE.exists():
    raise SystemExit(f"missing UGA still: {IMAGE}")

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url=os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
    api_key=api_key,
).configure(InferenceConfiguration(
    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="andre-4cotb",
    workflow_id="defensive-coverage-analysis-1789781275490",
    images={
        "image": str(IMAGE),  # Path to your image file
    },
    use_cache=True,  # Speeds up repeated requests
)

# 4. Get your results — image blobs are written to disk so stdout stays readable.
payload = result[0] if isinstance(result, list) and result else result
if not isinstance(payload, dict):
    print(result)
    raise SystemExit("unexpected workflow result shape")

compact: dict = {}
for key, value in payload.items():
    if key in {"minimap", "output_image"}:
        path = OUT / f"{key}.png"
        compact[key] = str(path) if _write_image(value, path) else None
        continue
    compact[key] = value

print(json.dumps(compact, indent=2, default=str))
