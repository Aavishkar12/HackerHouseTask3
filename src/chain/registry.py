"""
registry.py — Stage 3: put a Stage 2 match record on a blockchain and
verify it again later.

What actually goes on chain
---------------------------
Only a 32-byte fingerprint: SHA-256 over the canonical JSON payload that
`search.record.MatchRecord.hashed_payload()` produces. No image, no URL,
no name, no personal data of any kind. The chain is used as a tamper-
evident timestamp, not as storage.

Verification is therefore a pure recomputation:

    load match_record.json -> rebuild canonical payload -> sha256
        -> look the hash up on chain
        -> present  = the record is byte-for-byte what was anchored
        -> absent   = something in the record changed (or was never anchored)

Where it runs
-------------
Anything that speaks JSON-RPC: a local Ganache/Anvil/Hardhat node, or a
public testnet (Sepolia, Polygon Amoy). Two signing modes:

    node account   no key needed — uses the dev node's first unlocked
                   account. This is the default and is what local demos use.
    local key      set PRIVATE_KEY; required for public testnets, where
                   the node will not sign for you.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_PATH = REPO_ROOT / "contracts" / "MatchRegistry.json"
DEPLOYMENT_PATH = REPO_ROOT / "data" / "output" / "deployment.json"

DEFAULT_RPC = "http://127.0.0.1:8545"

# Chain ids we can name in output. Anything else is reported by number.
KNOWN_CHAINS = {
    1: "Ethereum mainnet",
    11155111: "Sepolia testnet",
    137: "Polygon mainnet",
    80002: "Polygon Amoy testnet",
    1337: "local development chain",
    31337: "local development chain",
}

# Chains that are NOT test/local environments. Refused by default so a
# demo key can never accidentally spend real funds.
MAINNET_CHAIN_IDS = {1, 137, 56, 43114, 10, 42161, 8453}


class ChainError(Exception):
    """Anything that stops us talking to the chain or the contract."""


def load_env_file(path: Path | str | None = None) -> dict:
    """Read a .env file into os.environ (without overwriting real env
    vars). Deliberately tiny — no python-dotenv dependency for what is
    three settings.

    Returns the keys it set, so callers can report what was loaded
    WITHOUT ever printing a value (PRIVATE_KEY lives in here).
    """
    p = Path(path) if path else REPO_ROOT / ".env"
    loaded = {}
    if not p.exists():
        return loaded

    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key not in os.environ:
            os.environ[key] = value
        loaded[key] = True
    return loaded


# --------------------------------------------------------------------------
# artifact
# --------------------------------------------------------------------------

def load_artifact(path: Path | str | None = None) -> dict:
    """Load the precompiled ABI + bytecode committed in contracts/."""
    p = Path(path) if path else ARTIFACT_PATH
    if not p.exists():
        raise ChainError(
            f"Contract artifact not found: {p}\n"
            "    Run: python scripts/compile_contract.py")
    try:
        artifact = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ChainError(f"Contract artifact is not valid JSON: {e}") from e

    for key in ("abi", "bytecode"):
        if key not in artifact:
            raise ChainError(f"Contract artifact is missing '{key}': {p}")
    return artifact


def normalize_hash(content_hash: str) -> str:
    """Accept a sha256 hex digest with or without 0x; return 0x-prefixed."""
    if not isinstance(content_hash, str):
        raise ChainError(f"Hash must be a hex string, got {type(content_hash).__name__}")
    h = content_hash.strip().lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ChainError(
            f"Expected a 32-byte (64 hex char) hash, got {len(h)} chars: {h[:20]}...")
    try:
        int(h, 16)
    except ValueError as e:
        raise ChainError(f"Hash is not valid hex: {content_hash}") from e
    if int(h, 16) == 0:
        raise ChainError("Hash is all zeros — the contract rejects that "
                         "(it is the 'never anchored' sentinel).")
    return "0x" + h


# --------------------------------------------------------------------------
# connection
# --------------------------------------------------------------------------

@dataclass
class ChainConnection:
    w3: Any
    address: str
    chain_id: int
    rpc_url: str
    account: Any = None          # LocalAccount when signing locally
    simulated: bool = False

    @property
    def chain_name(self) -> str:
        if self.simulated:
            return "in-process simulated chain"
        return KNOWN_CHAINS.get(self.chain_id, f"chain id {self.chain_id}")

    @property
    def signing_mode(self) -> str:
        if self.simulated:
            return "in-process EVM (eth-tester)"
        return "local private key" if self.account else "node account (unlocked)"

    def balance_eth(self) -> float:
        wei = self.w3.eth.get_balance(self.address)
        return wei / 1e18

    def describe(self) -> str:
        where = "in-process EVM" if self.simulated else self.rpc_url
        return (f"{self.chain_name} via {where}\n"
                f"    account: {self.address} "
                f"({self.balance_eth():.4f} native token)\n"
                f"    signing: {self.signing_mode}")

    # -- transaction plumbing ------------------------------------------
    def send(self, tx: dict) -> Any:
        """Sign (if we hold a key) and send a built transaction; wait for
        the receipt. Returns the receipt."""
        w3 = self.w3
        tx.setdefault("from", self.address)
        tx.setdefault("nonce", w3.eth.get_transaction_count(self.address))
        tx.setdefault("chainId", self.chain_id)

        if "gas" not in tx:
            try:
                tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.25)
            except Exception:
                tx["gas"] = 300_000

        if self.account is not None:
            signed = self.account.sign_transaction(tx)
            # web3 v7+ exposes .raw_transaction; older versions .rawTransaction
            raw = getattr(signed, "raw_transaction", None)
            if raw is None:
                raw = signed.rawTransaction
            tx_hash = w3.eth.send_raw_transaction(raw)
        else:
            tx_hash = w3.eth.send_transaction(tx)

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status != 1:
            raise ChainError(
                f"Transaction reverted on chain (tx {_hex(receipt.transactionHash)})")
        return receipt


def connect(rpc_url: Optional[str] = None,
            private_key: Optional[str] = None,
            simulated: bool = False,
            allow_mainnet: bool = False) -> ChainConnection:
    """Open a connection to a chain.

    rpc_url      defaults to $RPC_URL, then http://127.0.0.1:8545
    private_key  defaults to $PRIVATE_KEY; if absent, the node's first
                 unlocked account is used (dev nodes only)
    simulated    ignore the RPC entirely and run an in-process EVM
    """
    try:
        from web3 import Web3
    except ImportError as e:
        raise ChainError(
            "web3 is not installed.\n"
            "    pip install -r requirements.txt") from e

    if simulated:
        return _connect_simulated(Web3)

    rpc_url = rpc_url or os.environ.get("RPC_URL") or DEFAULT_RPC
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))

    if not w3.is_connected():
        raise ChainError(
            f"No JSON-RPC node answering at {rpc_url}\n"
            "    Start a local chain:   npx ganache --wallet.deterministic\n"
            "    or point RPC_URL at a testnet,\n"
            "    or run with --simulated to use the in-process EVM.")

    chain_id = w3.eth.chain_id

    # Proof-of-authority chains (Polygon Amoy and friends) put extra data
    # in block headers that the default validator rejects.
    if chain_id not in (1337, 31337):
        _install_poa_middleware(w3)

    if chain_id in MAINNET_CHAIN_IDS and not allow_mainnet:
        raise ChainError(
            f"{KNOWN_CHAINS.get(chain_id, chain_id)} is a MAINNET chain and "
            "real funds are at risk.\n"
            "    This project is a prototype — use a local node or a testnet.\n"
            "    Pass --allow-mainnet only if you truly mean it.")

    private_key = private_key or os.environ.get("PRIVATE_KEY") or ""
    private_key = private_key.strip()

    if private_key:
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        try:
            account = w3.eth.account.from_key(private_key)
        except Exception as e:
            raise ChainError(f"PRIVATE_KEY is not a valid key: {e}") from e
        conn = ChainConnection(w3=w3, address=account.address,
                               chain_id=chain_id, rpc_url=rpc_url,
                               account=account)
    else:
        accounts = w3.eth.accounts
        if not accounts:
            raise ChainError(
                "The node has no unlocked accounts and no PRIVATE_KEY is set.\n"
                "    Public testnets never unlock accounts for you — set "
                "PRIVATE_KEY in .env (a funded TESTNET key).")
        conn = ChainConnection(w3=w3, address=accounts[0],
                               chain_id=chain_id, rpc_url=rpc_url)

    if conn.balance_eth() == 0:
        raise ChainError(
            f"Account {conn.address} has a zero balance on "
            f"{conn.chain_name} — it cannot pay gas.\n"
            "    Fund it from that network's faucet, or use a local node.")
    return conn


def _install_poa_middleware(w3) -> None:
    """Best-effort POA support; harmless on non-POA chains."""
    try:  # web3 v7/v8
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return
    except Exception:
        pass
    try:  # web3 v6
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    except Exception:
        pass


def _connect_simulated(Web3) -> ChainConnection:
    """In-process EVM. No node, no install, no network — but the chain
    only exists for the lifetime of this process."""
    try:
        from web3 import EthereumTesterProvider
    except ImportError as e:
        raise ChainError("This web3 build has no EthereumTesterProvider.") from e

    try:
        w3 = Web3(EthereumTesterProvider())
    except ImportError as e:
        raise ChainError(
            "Simulated mode needs the eth-tester extra:\n"
            "    pip install \"web3[tester]\"\n"
            "Or run a real local node instead: npx ganache") from e

    return ChainConnection(w3=w3, address=w3.eth.accounts[0],
                           chain_id=w3.eth.chain_id, rpc_url="(in-process)",
                           simulated=True)


# --------------------------------------------------------------------------
# contract operations
# --------------------------------------------------------------------------

def get_contract(conn: ChainConnection, address: str):
    artifact = load_artifact()
    try:
        checksum = conn.w3.to_checksum_address(address)
    except Exception as e:
        raise ChainError(f"Not a valid contract address: {address}") from e

    code = conn.w3.eth.get_code(checksum)
    if not code or code in (b"", b"0x", "0x"):
        raise ChainError(
            f"No contract deployed at {checksum} on {conn.chain_name}.\n"
            "    Wrong address, or the chain was restarted (local nodes "
            "forget everything on restart — redeploy).")

    return conn.w3.eth.contract(address=checksum, abi=artifact["abi"])


def _hex(value) -> str:
    """web3 v7+ returns bare hex from .hex(); block explorers want 0x."""
    h = value.hex() if hasattr(value, "hex") else str(value)
    return h if h.startswith("0x") else "0x" + h


def deploy(conn: ChainConnection) -> dict:
    """Deploy MatchRegistry. Returns deployment details."""
    artifact = load_artifact()
    factory = conn.w3.eth.contract(abi=artifact["abi"],
                                   bytecode=artifact["bytecode"])
    tx = factory.constructor().build_transaction({
        "from": conn.address,
        "nonce": conn.w3.eth.get_transaction_count(conn.address),
        "chainId": conn.chain_id,
    })
    tx.pop("to", None)
    receipt = conn.send(tx)

    return {
        "contract_address": receipt.contractAddress,
        "chain_id": conn.chain_id,
        "chain_name": conn.chain_name,
        "rpc_url": conn.rpc_url,
        "deployer": conn.address,
        "tx_hash": _hex(receipt.transactionHash),
        "block_number": receipt.blockNumber,
        "gas_used": receipt.gasUsed,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "solc_version": artifact.get("solcVersion", "unknown"),
    }


@dataclass
class AnchorResult:
    content_hash: str
    tx_hash: str
    block_number: int
    gas_used: int
    timestamp: int
    already_anchored: bool = False
    note: str = ""


def anchor_hash(conn: ChainConnection, contract_address: str,
                content_hash: str) -> AnchorResult:
    """Write a fingerprint on chain. Anchoring the same hash twice is
    treated as success — the original record stands, which is the point."""
    h = normalize_hash(content_hash)
    contract = get_contract(conn, contract_address)

    exists, ts, _submitter = contract.functions.verify(h).call()
    if exists:
        return AnchorResult(
            content_hash=h, tx_hash="", block_number=0, gas_used=0,
            timestamp=ts, already_anchored=True,
            note="This exact record was already anchored; the original "
                 "timestamp is immutable and was kept.")

    tx = contract.functions.anchor(h).build_transaction({
        "from": conn.address,
        "nonce": conn.w3.eth.get_transaction_count(conn.address),
        "chainId": conn.chain_id,
    })
    receipt = conn.send(tx)
    block = conn.w3.eth.get_block(receipt.blockNumber)

    return AnchorResult(
        content_hash=h,
        tx_hash=_hex(receipt.transactionHash),
        block_number=receipt.blockNumber,
        gas_used=receipt.gasUsed,
        timestamp=block.timestamp,
    )


@dataclass
class VerifyResult:
    content_hash: str
    found: bool
    timestamp: int = 0
    submitter: str = ""
    chain_name: str = ""
    contract_address: str = ""

    @property
    def anchored_at(self) -> str:
        if not self.found:
            return ""
        return datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc).isoformat()

    @property
    def summary(self) -> str:
        if self.found:
            return (f"VERIFIED — anchored {self.anchored_at} "
                    f"by {self.submitter}")
        return "NOT ON CHAIN — this exact record was never anchored"


def verify_hash(conn: ChainConnection, contract_address: str,
                content_hash: str) -> VerifyResult:
    """Look a fingerprint up on chain. This is a read — no gas, no key."""
    h = normalize_hash(content_hash)
    contract = get_contract(conn, contract_address)
    exists, ts, submitter = contract.functions.verify(h).call()

    return VerifyResult(
        content_hash=h,
        found=bool(exists),
        timestamp=int(ts),
        submitter=submitter if exists else "",
        chain_name=conn.chain_name,
        contract_address=contract.address,
    )


def total_anchored(conn: ChainConnection, contract_address: str) -> int:
    return get_contract(conn, contract_address).functions.total().call()


# --------------------------------------------------------------------------
# deployment bookkeeping
# --------------------------------------------------------------------------

def save_deployment(info: dict, path: Path | str | None = None) -> Path:
    p = Path(path) if path else DEPLOYMENT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2) + "\n")
    return p


def load_deployment(path: Path | str | None = None) -> Optional[dict]:
    p = Path(path) if path else DEPLOYMENT_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def resolve_contract_address(explicit: Optional[str] = None,
                             deployment_path: Path | str | None = None
                             ) -> Optional[str]:
    """--contract flag, else $CONTRACT_ADDRESS, else the last deployment."""
    if explicit:
        return explicit
    env = os.environ.get("CONTRACT_ADDRESS", "").strip()
    if env:
        return env
    dep = load_deployment(deployment_path)
    if dep:
        return dep.get("contract_address")
    return None
