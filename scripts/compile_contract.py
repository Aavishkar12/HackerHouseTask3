#!/usr/bin/env python3
"""
compile_contract.py — compile contracts/MatchRegistry.sol into a
precompiled artifact (ABI + bytecode) checked into the repo.

You normally do NOT need to run this. contracts/MatchRegistry.json is
committed, so deploying works offline with no Solidity toolchain
installed. Run this only if you change the .sol source.

    python scripts/compile_contract.py

Two compilers are tried, in order:
    1. py-solc-x  (pip install py-solc-x) — downloads a pinned solc once
    2. npm solc   (npm install solc@0.8.24) — used if the solc binary
       download is blocked by a proxy/firewall

The committed artifact was built with route 2, solc 0.8.24, optimizer on
at 200 runs — identical settings either way.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOL_PATH = REPO_ROOT / "contracts" / "MatchRegistry.sol"
OUT_PATH = REPO_ROOT / "contracts" / "MatchRegistry.json"

SOLC_VERSION = "0.8.24"


STANDARD_INPUT = {
    "language": "Solidity",
    "sources": {"MatchRegistry.sol": {"content": None}},  # filled at runtime
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}},
    },
}


def _standard_input(source: str) -> dict:
    payload = json.loads(json.dumps(STANDARD_INPUT))
    payload["sources"]["MatchRegistry.sol"]["content"] = source
    return payload


def compile_with_solcx(source: str):
    """Route 1: py-solc-x. Returns the contract dict, or None if unavailable."""
    try:
        import solcx
    except ImportError:
        print("[i] py-solc-x not installed — skipping.")
        return None

    try:
        installed = [str(v) for v in solcx.get_installed_solc_versions()]
        if SOLC_VERSION not in installed:
            print(f"[*] Downloading solc {SOLC_VERSION} (one-time) ...")
            solcx.install_solc(SOLC_VERSION)
        print(f"[*] Compiling with py-solc-x / solc {SOLC_VERSION} ...")
        out = solcx.compile_standard(
            _standard_input(source), solc_version=SOLC_VERSION)
    except Exception as e:  # download blocked, version unavailable, etc.
        print(f"[i] py-solc-x route failed ({type(e).__name__}) — "
              "falling back to npm solc.")
        return None

    return out["contracts"]["MatchRegistry.sol"]["MatchRegistry"]


def compile_with_npm_solc(source: str):
    """Route 2: the npm `solc` package. Returns the contract dict or None."""
    if shutil.which("node") is None:
        print("[i] node not found — cannot use the npm solc route.")
        return None

    node_modules = REPO_ROOT / "node_modules" / "solc"
    if not node_modules.exists():
        print(f"[*] Installing npm solc@{SOLC_VERSION} (one-time) ...")
        r = subprocess.run(
            ["npm", "install", "--no-save", f"solc@{SOLC_VERSION}"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[!] npm install failed:\n{r.stderr[-800:]}")
            return None

    driver = """
    const solc = require(process.argv[2]);
    const fs = require('fs');
    const input = fs.readFileSync(process.argv[3], 'utf8');
    const out = JSON.parse(solc.compile(input));
    for (const e of (out.errors || [])) {
        console.error(e.severity.toUpperCase() + ': ' + e.formattedMessage);
    }
    if ((out.errors || []).some(e => e.severity === 'error')) process.exit(1);
    fs.writeFileSync(process.argv[4], JSON.stringify(out));
    """

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "driver.js").write_text(driver)
        (tmp / "in.json").write_text(json.dumps(_standard_input(source)))
        print(f"[*] Compiling with npm solc {SOLC_VERSION} ...")
        r = subprocess.run(
            ["node", str(tmp / "driver.js"), str(node_modules),
             str(tmp / "in.json"), str(tmp / "out.json")],
            capture_output=True, text=True)
        if r.stderr.strip():
            print(r.stderr.strip())
        if r.returncode != 0:
            return None
        out = json.loads((tmp / "out.json").read_text())

    return out["contracts"]["MatchRegistry.sol"]["MatchRegistry"]


def main() -> int:
    if not SOL_PATH.exists():
        print(f"[!] Contract source not found: {SOL_PATH}")
        return 1

    source = SOL_PATH.read_text()
    contract = compile_with_solcx(source) or compile_with_npm_solc(source)

    if contract is None:
        print("\n[!] Could not compile with either route.")
        print("    You do not need to compile to run this project — "
              "contracts/MatchRegistry.json is already committed.")
        return 1

    artifact = {
        "contractName": "MatchRegistry",
        "solcVersion": SOLC_VERSION,
        "optimizer": {"enabled": True, "runs": 200},
        "abi": contract["abi"],
        "bytecode": "0x" + contract["evm"]["bytecode"]["object"],
    }

    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    size = len(artifact["bytecode"]) // 2 - 1
    print(f"[+] Wrote {OUT_PATH} ({size} bytes of deploy bytecode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
