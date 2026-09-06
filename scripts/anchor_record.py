#!/usr/bin/env python3
"""
anchor_record.py — Stage 3 step 2: put the Stage 2 match record on chain.

What gets written
-----------------
One 32-byte value: SHA-256 over the canonical payload of the match record
Stage 2 produced. No image, no URL, no name — see contracts/MatchRegistry.sol
for why that is deliberate.

The fields that go into the hash are the substantive claims (which post
was found, on which platform, whether the face verified, who was
identified). Volatile things — timestamps, local file paths, result
counts — are excluded, so the same discovery hashes identically on any
machine. `src/search/record.py::hashed_payload` is the definition.

Usage
-----
    # normal: a node is running, contract already deployed
    python scripts/anchor_record.py

    # everything in one process, no node, no install (chain is ephemeral)
    python scripts/anchor_record.py --simulated

    # explicit inputs
    python scripts/anchor_record.py --record data/output/match_record.json \
        --contract 0xabc... --rpc http://127.0.0.1:8545

Output
------
    data/output/anchor_receipt.json   (what verify_onchain.py reads back)
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chain.registry import (  # noqa: E402
    ChainError,
    anchor_hash,
    connect,
    deploy,
    load_env_file,
    resolve_contract_address,
    save_deployment,
    total_anchored,
    verify_hash,
)
from search.record import MatchRecord  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "output"
RECORD_PATH = OUTPUT_DIR / "match_record.json"
RECEIPT_PATH = OUTPUT_DIR / "anchor_receipt.json"
# A simulated run's contract address is dead the moment the process exits,
# so it must never overwrite a receipt pointing at a real chain.
SIM_RECEIPT_PATH = OUTPUT_DIR / "anchor_receipt_simulated.json"

BAR = "=" * 68


def banner(step: str, title: str) -> None:
    print(f"\n{BAR}\n  {step} — {title}\n{BAR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default=str(RECORD_PATH),
                    help=f"Stage 2 match record (default: {RECORD_PATH.name})")
    ap.add_argument("--contract",
                    help="contract address (default: $CONTRACT_ADDRESS, "
                         "else the last deployment)")
    ap.add_argument("--rpc", help="JSON-RPC URL (default: $RPC_URL, "
                                  "else http://127.0.0.1:8545)")
    ap.add_argument("--simulated", action="store_true",
                    help="in-process EVM: deploy + anchor + verify in one run")
    ap.add_argument("--allow-mainnet", action="store_true")
    args = ap.parse_args()

    load_env_file()

    # ----------------------------------------------------------- 1. the hash
    banner("STEP 1/3", "hash the Stage 2 match record")

    record_path = Path(args.record)
    if not record_path.exists():
        print(f"[!] No match record at {record_path}")
        print("    Run Stage 2 first: python scripts/find_match.py")
        return 1

    try:
        record = MatchRecord.load(record_path)
    except Exception as e:
        print(f"[!] Could not read the match record: {type(e).__name__}: {e}")
        return 1

    content_hash = record.content_hash()

    print(f"[*] Record   : {record_path}")
    print(f"    subject  : {record.identified_subject or 'UNKNOWN'}")
    print(f"    post     : {record.post_url}")
    print(f"    platform : {record.platform}")
    print(f"    face ok  : {record.face_verified}")
    print(f"\n[*] Canonical payload actually hashed:")
    for k, v in record.hashed_payload().items():
        shown = str(v)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        print(f"      {k:<26} {shown}")
    print(f"\n[+] SHA-256 content hash: {content_hash}")

    # ------------------------------------------------------------- 2. connect
    banner("STEP 2/3", "connect to the chain")
    try:
        conn = connect(rpc_url=args.rpc, simulated=args.simulated,
                       allow_mainnet=args.allow_mainnet)
        print(f"[*] {conn.describe()}")

        address = resolve_contract_address(args.contract)

        if args.simulated:
            # The in-process chain is brand new every run, so whatever
            # address a previous run saved is meaningless here.
            print("\n[*] Simulated chain is empty — deploying the contract "
                  "into it ...")
            info = deploy(conn)
            address = info["contract_address"]
            print(f"[+] Deployed at {address} (block {info['block_number']})")
        elif not address:
            print("\n[!] No contract address available.")
            print("    Run: python scripts/deploy_contract.py")
            print("    (or pass --contract 0x..., or set CONTRACT_ADDRESS)")
            return 1
        else:
            print(f"[*] Contract: {address}")

        # -------------------------------------------------------- 3. anchor
        banner("STEP 3/3", "anchor the hash on chain")
        result = anchor_hash(conn, address, content_hash)

        if result.already_anchored:
            print(f"[i] {result.note}")
        else:
            print(f"[+] Anchored in transaction {result.tx_hash}")
            print(f"    block   : {result.block_number}")
            print(f"    gas used: {result.gas_used}")

        anchored_at = datetime.fromtimestamp(
            result.timestamp, tz=timezone.utc).isoformat()
        print(f"    on-chain timestamp: {anchored_at}")

        # Read it straight back, so the run proves the write landed rather
        # than trusting the receipt.
        check = verify_hash(conn, address, content_hash)
        print(f"\n[*] Immediate read-back: {check.summary}")
        if not check.found:
            print("[!] The hash is not readable on chain after anchoring — "
                  "something is wrong.")
            return 1
        print(f"[*] Records anchored in this contract: "
              f"{total_anchored(conn, address)}")

    except ChainError as e:
        print(f"\n[!] {e}")
        return 1

    # ------------------------------------------------------------- receipt
    receipt = {
        "content_hash": result.content_hash,
        "record_path": str(record_path),
        "record_version": record.record_version,
        "post_url": record.post_url,
        "identified_subject": record.identified_subject,
        "contract_address": address,
        "chain_id": conn.chain_id,
        "chain_name": conn.chain_name,
        "rpc_url": conn.rpc_url,
        "simulated": conn.simulated,
        "tx_hash": result.tx_hash,
        "block_number": result.block_number,
        "gas_used": result.gas_used,
        "anchored_at_utc": anchored_at,
        "anchored_by": check.submitter,
        "already_anchored": result.already_anchored,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = SIM_RECEIPT_PATH if args.simulated else RECEIPT_PATH
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    if args.simulated:
        # verify_onchain.py cannot reach this chain (it dies with this
        # process), so prove tamper-detection here instead — otherwise
        # simulated mode would only ever show the happy path.
        banner("EXTRA", "tamper check (simulated mode runs it inline)")
        original_url = record.post_url
        forged = replace(record, post_url=original_url + "?edited")
        forged_hash = forged.content_hash()
        print(f"[*] Same record with post_url edited to "
              f"'{original_url[:40]}...?edited'")
        print(f"    recomputed hash: {forged_hash}")
        forged_check = verify_hash(conn, address, forged_hash)
        print(f"[*] On-chain lookup: {forged_check.summary}")
        if forged_check.found:
            print("[!] A tampered record verified — that must never happen.")
            return 1
        print("[+] Tamper detected: the edited record does not match the "
              "on-chain fingerprint.")

        print("\n[!] Simulated chain — this contract and anchor vanish when "
              "the script exits.")
        print("    Nothing was written to deployment.json.")

    print(f"\n{BAR}\n  RESULT\n{BAR}")
    print(f"  Content hash : {result.content_hash}")
    print(f"  Chain        : {conn.chain_name}")
    print(f"  Contract     : {address}")
    print(f"  Anchored at  : {anchored_at}")
    print(f"\n[+] Receipt -> {receipt_path}")
    if args.simulated:
        print("    Verification already ran above — verify_onchain.py cannot "
              "reach\n    this chain, because it no longer exists. Run a real "
              "node for a\n    separate verification step.")
    else:
        print("    Verify it any time with: python scripts/verify_onchain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
