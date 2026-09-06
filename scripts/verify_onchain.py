#!/usr/bin/env python3
"""
verify_onchain.py — Stage 3 step 3: re-verify the match record against
the blockchain.

This is the step the task actually asks for: prove that the data found in
Stage 2 is the same data that was anchored, using only what is on chain.

How the proof works
-------------------
Nothing is trusted from the receipt file except the contract address.
The hash is RECOMPUTED from match_record.json every time:

    match_record.json -> canonical payload -> SHA-256 -> lookup on chain

    found     -> the record is byte-for-byte identical to what was
                 anchored, and the chain says when
    not found -> either it was never anchored, or a hashed field changed

Because the hash is recomputed rather than read back, editing the record
cannot pass verification — which is what `--tamper` demonstrates.

Usage
-----
    # verify the current record against the chain
    python scripts/verify_onchain.py

    # prove that editing the record breaks verification
    python scripts/verify_onchain.py --tamper

    # verify a bare hash without any local file
    python scripts/verify_onchain.py --hash 0x1234...

Exit codes
----------
    0  verified on chain
    1  could not run (no node, no contract, no record)
    2  ran fine, but the record is NOT on chain (tampered or never anchored)
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chain.registry import (  # noqa: E402
    ChainError,
    connect,
    load_env_file,
    normalize_hash,
    resolve_contract_address,
    total_anchored,
    verify_hash,
)
from search.record import MatchRecord  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "output"
RECORD_PATH = OUTPUT_DIR / "match_record.json"
RECEIPT_PATH = OUTPUT_DIR / "anchor_receipt.json"

BAR = "=" * 68


def banner(step: str, title: str) -> None:
    print(f"\n{BAR}\n  {step} — {title}\n{BAR}")


def address_from_receipt() -> tuple[str | None, str | None]:
    """Fall back to the anchor receipt for contract address and RPC."""
    if not RECEIPT_PATH.exists():
        return None, None
    try:
        data = json.loads(RECEIPT_PATH.read_text())
    except json.JSONDecodeError:
        return None, None
    rpc = data.get("rpc_url")
    if rpc == "(in-process)":
        rpc = None
    return data.get("contract_address"), rpc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default=str(RECORD_PATH),
                    help=f"match record to verify (default: {RECORD_PATH.name})")
    ap.add_argument("--hash", dest="raw_hash",
                    help="verify this hash directly, ignoring local files")
    ap.add_argument("--contract",
                    help="contract address (default: last deployment / receipt)")
    ap.add_argument("--rpc", help="JSON-RPC URL")
    ap.add_argument("--tamper", action="store_true",
                    help="also show that an edited record fails verification")
    ap.add_argument("--simulated", action="store_true",
                    help="(not supported here — explains why)")
    ap.add_argument("--allow-mainnet", action="store_true")
    args = ap.parse_args()

    if args.simulated:
        # connect() suggests --simulated on failure, so people will try it
        # here. An argparse "unrecognized argument" mid-demo is worse than
        # a straight answer.
        print("[!] --simulated cannot work for verification.")
        print("    The in-process chain exists only while a single script "
              "runs, so\n    by the time this process starts, the chain that "
              "held the anchor is\n    already gone.")
        print("\n    For a self-contained simulated demo (anchor AND verify "
              "in one run):")
        print("        python scripts/anchor_record.py --simulated")
        print("\n    For a real two-step demo, start a node:")
        print("        npx ganache --wallet.deterministic")
        return 1

    load_env_file()

    print(f"{BAR}\n  STAGE 3 — verify against the blockchain\n{BAR}")

    # ------------------------------------------------------ 1. recompute
    record = None
    if args.raw_hash:
        try:
            content_hash = normalize_hash(args.raw_hash)
        except ChainError as e:
            print(f"[!] {e}")
            return 1
        print(f"\n[*] Verifying a hash supplied directly: {content_hash}")
    else:
        record_path = Path(args.record)
        if not record_path.exists():
            print(f"\n[!] No match record at {record_path}")
            print("    Run Stage 2 first: python scripts/find_match.py")
            return 1
        try:
            record = MatchRecord.load(record_path)
        except Exception as e:
            print(f"\n[!] Could not read the record: {type(e).__name__}: {e}")
            return 1

        banner("STEP 1/2", "recompute the hash from the record on disk")
        content_hash = record.content_hash()
        print(f"[*] Record  : {record_path}")
        print(f"    subject : {record.identified_subject or 'UNKNOWN'}")
        print(f"    post    : {record.post_url}")
        print(f"\n[+] Recomputed SHA-256: {content_hash}")
        print("    (recomputed from the file right now — not read from the "
              "anchor receipt)")

    # --------------------------------------------------------- 2. lookup
    banner("STEP 2/2", "look the hash up on chain")

    receipt_addr, receipt_rpc = address_from_receipt()
    address = resolve_contract_address(args.contract) or receipt_addr
    rpc = args.rpc or receipt_rpc

    if not address:
        print("[!] No contract address known.")
        print("    Deploy first (scripts/deploy_contract.py) or pass "
              "--contract 0x...")
        return 1

    try:
        conn = connect(rpc_url=rpc, allow_mainnet=args.allow_mainnet)
        print(f"[*] {conn.describe()}")
        print(f"[*] Contract: {address}")
        result = verify_hash(conn, address, content_hash)
        anchored_total = total_anchored(conn, address)
    except ChainError as e:
        print(f"\n[!] {e}")
        return 1

    print(f"\n[*] Records anchored in this contract: {anchored_total}")
    print(f"\n{BAR}")
    if result.found:
        print(f"  ✓ {result.summary}")
        print(f"{BAR}")
        print(f"  hash        : {result.content_hash}")
        print(f"  anchored at : {result.anchored_at}  (block time, UTC)")
        print(f"  anchored by : {result.submitter}")
        print(f"  chain       : {result.chain_name}")
        print(f"  contract    : {result.contract_address}")
        print("\n  The record on disk is byte-for-byte the record that was "
              "anchored.")
    else:
        print(f"  ✗ {result.summary}")
        print(f"{BAR}")
        print(f"  hash : {result.content_hash}")
        print("\n  Either this record was never anchored, or one of the "
              "hashed fields\n  has changed since it was. Both are real "
              "outcomes, not errors.")

    # ---------------------------------------------------- tamper evidence
    if args.tamper:
        if record is None:
            print("\n[!] --tamper needs a record file (not --hash).")
            return 0 if result.found else 2

        banner("TAMPER CHECK", "edit one field and verify again")
        original = record.post_url
        forged = replace(record, post_url=original + "?edited")
        forged_hash = forged.content_hash()

        print(f"[*] Original post_url : {original}")
        print(f"[*] Edited  post_url : {original}?edited")
        print(f"\n    original hash : {content_hash}")
        print(f"    edited   hash : {forged_hash}")
        print("    (one character changed in the record -> a completely "
              "different hash)")

        try:
            forged_result = verify_hash(conn, address, forged_hash)
        except ChainError as e:
            print(f"[!] {e}")
            return 1

        print(f"\n[*] On-chain lookup of the edited record: "
              f"{forged_result.summary}")
        if forged_result.found:
            print("[!] A tampered record verified — that must never happen.")
            return 1
        print("\n[+] Tamper detected. The blockchain accepts only the exact "
              "record\n    that was anchored; any edit produces a hash that "
              "is not on chain.")

    if not result.found:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
