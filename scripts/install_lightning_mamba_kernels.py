"""Install and verify the exact CUDA Mamba wheels used on Lightning H100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch


WHEELS = (
    {
        "distribution": "causal-conv1d",
        "version": "1.6.2.post1",
        "filename": "causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "url": "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "sha256": "ee5a87ebc4a8bb04408e20cc95cdb85e90742b90c0bcdfb1384a88758f2ee246",
    },
    {
        "distribution": "mamba-ssm",
        "version": "2.3.2.post1",
        "filename": "mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "url": "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "sha256": "eabd2bfadeda8eeab6b14bdeb4486ccacd41174c02ffc1b527808d3e2b7e19ec",
    },
)


def _validate_environment() -> None:
    torch_base = torch.__version__.split("+")[0]
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Lightning kernel wheels require Python 3.12, found {sys.version}"
        )
    if torch_base != "2.8.0":
        raise RuntimeError(
            f"Lightning kernel wheels require torch 2.8.0, found {torch.__version__}"
        )
    if not str(torch.version.cuda).startswith("12."):
        raise RuntimeError(
            f"Lightning kernel wheels require CUDA 12.x torch, found {torch.version.cuda}"
        )
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            f"Lightning kernel wheels require Linux x86_64, found {platform.system()} {platform.machine()}"
        )
    if not bool(torch.compiled_with_cxx11_abi()):
        raise RuntimeError("Lightning torch must use the CXX11 ABI")


def _installed_wheels_are_exact() -> bool:
    try:
        for item in WHEELS:
            distribution = metadata.distribution(item["distribution"])
            if distribution.version.lower() != item["version"]:
                return False
            direct_url_text = distribution.read_text("direct_url.json")
            if direct_url_text is None:
                return False
            direct_url = json.loads(direct_url_text)
            recorded_hash = direct_url.get("archive_info", {}).get("hash", "")
            if recorded_hash != f"sha256={item['sha256']}":
                return False
        return True
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return False


def _download(item: dict[str, str], destination: Path) -> Path:
    output = destination / item["filename"]
    request = urllib.request.Request(
        item["url"], headers={"User-Agent": "SpectralGuard-Phase3/3.0"}
    )
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(request, timeout=180) as response,
        output.open("wb") as handle,
    ):
        while block := response.read(1024 * 1024):
            digest.update(block)
            handle.write(block)
    observed = digest.hexdigest()
    if observed != item["sha256"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"wheel hash mismatch for {item['distribution']}: {observed} != {item['sha256']}"
        )
    return output


def _verify_fast_import() -> None:
    code = """
from transformers.models.mamba import modeling_mamba
assert modeling_mamba.is_fast_path_available, 'Transformers Mamba CUDA fast path is unavailable'
print('Mamba CUDA extension imports: PASS')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _verify_cuda_execution() -> None:
    code = """
import torch
from transformers import MambaConfig, MambaForCausalLM
from transformers.models.mamba import modeling_mamba
assert modeling_mamba.is_fast_path_available
assert torch.cuda.is_available()
config = MambaConfig(vocab_size=32, hidden_size=16, state_size=4, num_hidden_layers=2, expand=2, conv_kernel=3, time_step_rank=2, pad_token_id=0, eos_token_id=None)
model = MambaForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
input_ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
attention_mask = torch.ones_like(input_ids)
with torch.inference_mode():
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=2, eos_token_id=None, pad_token_id=0)
torch.cuda.synchronize()
assert torch.isfinite(output).all() and generated.shape[1] == 6
print('Mamba H100 fused forward and cached generation: PASS')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()
    _validate_environment()
    if not _installed_wheels_are_exact():
        with tempfile.TemporaryDirectory(prefix="spectralguard-mamba-") as temporary:
            wheels = [_download(item, Path(temporary)) for item in WHEELS]
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    *map(str, wheels),
                ],
                check=True,
            )
    if not _installed_wheels_are_exact():
        raise RuntimeError("installed Mamba wheels do not match the frozen hashes")
    _verify_fast_import()
    if args.cuda_smoke:
        _verify_cuda_execution()
    print("Lightning Mamba kernel setup: Finished")


if __name__ == "__main__":
    main()
