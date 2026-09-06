#!/usr/bin/env python3
"""
deploy_contract.py — Stage 3 step 1: deploy the MatchRegistry contract.

You only do this once per chain. The address is written to
data/output/deployment.json, so anchor_record.py and verify_onchain.py
pick it up automatically — you never have to copy-paste it.

Usage
-----
    # local dev chain (start one first: npx ganache --wallet.deterministic)
    python scripts/deploy_contract.py

    # no node installed at all — in-process EVM
    python scripts/deploy_contract.py --simulated

    # public testnet (needs RPC_URL + a funded PRIVATE_KEY in .env)
    python scripts/deploy_contract.py --rpc https://rpc-amoy.polygon.technology

NOTE on --simulated: the in-process chain exists only while this script
runs, so a contract deployed there is gone the moment it exits. Simulated
mode is only useful via `anchor_record.py --simulated`, which deploys,
anchors and verifies inside one process. For a real demo, run a node.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chain.registry import (  # noqa: E402
    ChainError,
    connect,
    deploy,
    load_env_file,
    save_deployment,
)

BAR = "=" * 68


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rpc", help="JSON-RPC URL (default: $RPC_URL, "
                                  "else http://127.0.0.1:8545)")
    ap.add_argument("--simulated", action="store_true",
                    help="in-process EVM instead of a real node")
    ap.add_argument("--allow-mainnet", action="store_true",
                    help="permit deploying to a mainnet (real funds!)")
    args = ap.parse_args()

    load_env_file()

    print(f"{BAR}\n  STAGE 3 — deploy MatchRegistry\n{BAR}")

    try:
        conn = connect(rpc_url=args.rpc, simulated=args.simulated,
                       allow_mainnet=args.allow_mainnet)
        print(f"[*] Connected: {conn.describe()}")

        print("\n[*] Deploying contract ...")
        info = deploy(conn)
    except ChainError as e:
        print(f"\n[!] {e}")
        return 1

    path = save_deployment(info)

    print(f"\n[+] Deployed MatchRegistry")
    print(f"    address : {info['contract_address']}")
    print(f"    chain   : {info['chain_name']} (id {info['chain_id']})")
    print(f"    tx      : {info['tx_hash']}")
    print(f"    block   : {info['block_number']}  gas: {info['gas_used']}")
    print(f"    solc    : {info['solc_version']}")
    print(f"\n[+] Saved deployment details -> {path}")
    print("    anchor_record.py and verify_onchain.py will read the address "
          "from there automatically.")

    if args.simulated:
        print("\n[!] This was the in-process chain — the contract is already "
              "gone.\n    Use `anchor_record.py --simulated` for a "
              "self-contained run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
