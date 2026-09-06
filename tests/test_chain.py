"""
Stage 3 tests.

The contract logic is exercised against a REAL EVM (eth-tester's
in-process py-evm), not a mock — a mocked chain would happily "verify"
anything and prove nothing. Tests that need it are skipped, not failed,
when eth-tester isn't installed, so the rest of the suite still runs.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chain import registry as R  # noqa: E402
from search.record import MatchRecord  # noqa: E402


# --------------------------------------------------------------------------
# hash normalisation — the boundary between Stage 2's hex and bytes32
# --------------------------------------------------------------------------

def test_normalize_accepts_bare_and_prefixed():
    h = "a" * 64
    assert R.normalize_hash(h) == "0x" + h
    assert R.normalize_hash("0x" + h) == "0x" + h


def test_normalize_is_case_insensitive():
    assert R.normalize_hash("A" * 64) == "0x" + "a" * 64


def test_normalize_strips_whitespace():
    assert R.normalize_hash("  " + "b" * 64 + "\n") == "0x" + "b" * 64


@pytest.mark.parametrize("bad", ["", "abc", "z" * 64, "a" * 63, "a" * 65])
def test_normalize_rejects_malformed(bad):
    with pytest.raises(R.ChainError):
        R.normalize_hash(bad)


def test_normalize_rejects_zero_hash():
    # bytes32(0) is the contract's "never anchored" sentinel; anchoring it
    # would make verification meaningless.
    with pytest.raises(R.ChainError):
        R.normalize_hash("0" * 64)


def test_normalize_rejects_non_string():
    with pytest.raises(R.ChainError):
        R.normalize_hash(12345)


# --------------------------------------------------------------------------
# compiled artifact
# --------------------------------------------------------------------------

def test_artifact_is_present_and_complete():
    a = R.load_artifact()
    assert a["contractName"] == "MatchRegistry"
    assert a["bytecode"].startswith("0x")
    assert len(a["bytecode"]) > 100
    names = {e.get("name") for e in a["abi"]}
    assert {"anchor", "verify", "total"} <= names


def test_artifact_missing_file_raises():
    with pytest.raises(R.ChainError):
        R.load_artifact(REPO_ROOT / "contracts" / "does_not_exist.json")


def test_artifact_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(R.ChainError):
        R.load_artifact(bad)


# --------------------------------------------------------------------------
# .env loading — must never overwrite a real environment variable
# --------------------------------------------------------------------------

def test_load_env_file_sets_missing_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('RPC_URL=http://example:8545\n# comment\n\nFOO="quoted"\n')
    monkeypatch.delenv("RPC_URL", raising=False)
    monkeypatch.delenv("FOO", raising=False)

    R.load_env_file(env)

    import os
    assert os.environ["RPC_URL"] == "http://example:8545"
    assert os.environ["FOO"] == "quoted"


def test_load_env_file_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RPC_URL=http://from-file:8545\n")
    monkeypatch.setenv("RPC_URL", "http://from-shell:8545")

    R.load_env_file(env)

    import os
    assert os.environ["RPC_URL"] == "http://from-shell:8545"


def test_load_env_file_missing_is_not_an_error(tmp_path):
    assert R.load_env_file(tmp_path / "nope.env") == {}


# --------------------------------------------------------------------------
# deployment bookkeeping
# --------------------------------------------------------------------------

def test_save_and_load_deployment_round_trip(tmp_path):
    p = tmp_path / "deployment.json"
    info = {"contract_address": "0x" + "1" * 40, "chain_id": 1337}
    R.save_deployment(info, p)
    assert R.load_deployment(p) == info


def test_load_deployment_missing_returns_none(tmp_path):
    assert R.load_deployment(tmp_path / "nope.json") is None


def test_load_deployment_corrupt_returns_none(tmp_path):
    p = tmp_path / "deployment.json"
    p.write_text("{broken")
    assert R.load_deployment(p) is None


def test_resolve_address_prefers_explicit(tmp_path, monkeypatch):
    p = tmp_path / "deployment.json"
    R.save_deployment({"contract_address": "0xFROM_FILE"}, p)
    monkeypatch.setenv("CONTRACT_ADDRESS", "0xFROM_ENV")
    assert R.resolve_contract_address("0xEXPLICIT", p) == "0xEXPLICIT"


def test_resolve_address_falls_back_to_env_then_file(tmp_path, monkeypatch):
    p = tmp_path / "deployment.json"
    R.save_deployment({"contract_address": "0xFROM_FILE"}, p)

    monkeypatch.setenv("CONTRACT_ADDRESS", "0xFROM_ENV")
    assert R.resolve_contract_address(None, p) == "0xFROM_ENV"

    monkeypatch.delenv("CONTRACT_ADDRESS")
    assert R.resolve_contract_address(None, p) == "0xFROM_FILE"


def test_resolve_address_none_when_nothing_known(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTRACT_ADDRESS", raising=False)
    assert R.resolve_contract_address(None, tmp_path / "nope.json") is None


# --------------------------------------------------------------------------
# against a real in-process EVM
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain():
    try:
        conn = R.connect(simulated=True)
    except R.ChainError as e:
        pytest.skip(f"in-process EVM unavailable: {e}")
    info = R.deploy(conn)
    return conn, info["contract_address"]


def test_deploy_produces_live_code(chain):
    conn, address = chain
    assert conn.w3.eth.get_code(address) not in (b"", b"0x")


def test_anchor_then_verify_round_trip(chain):
    conn, address = chain
    h = "1" * 64

    before = R.verify_hash(conn, address, h)
    assert before.found is False

    result = R.anchor_hash(conn, address, h)
    assert result.already_anchored is False
    assert result.block_number > 0

    after = R.verify_hash(conn, address, h)
    assert after.found is True
    assert after.timestamp == result.timestamp
    assert after.submitter.lower() == conn.address.lower()
    assert "VERIFIED" in after.summary


def test_unanchored_hash_is_not_found(chain):
    conn, address = chain
    result = R.verify_hash(conn, address, "9" * 64)
    assert result.found is False
    assert result.timestamp == 0
    assert "NOT ON CHAIN" in result.summary


def test_reanchoring_keeps_the_original_timestamp(chain):
    conn, address = chain
    h = "2" * 64

    first = R.anchor_hash(conn, address, h)
    second = R.anchor_hash(conn, address, h)

    assert second.already_anchored is True
    assert second.timestamp == first.timestamp
    assert second.tx_hash == ""  # no second transaction was sent


def test_total_counts_distinct_records(chain):
    conn, address = chain
    before = R.total_anchored(conn, address)
    R.anchor_hash(conn, address, "3" * 64)
    R.anchor_hash(conn, address, "3" * 64)   # duplicate, must not count
    R.anchor_hash(conn, address, "4" * 64)
    assert R.total_anchored(conn, address) == before + 2


def test_verify_against_address_with_no_contract(chain):
    conn, _ = chain
    with pytest.raises(R.ChainError, match="No contract deployed"):
        R.verify_hash(conn, "0x" + "de" * 20, "5" * 64)


def test_tampering_with_a_record_breaks_verification(chain, tmp_path):
    """The core Stage 3 claim, end to end."""
    conn, address = chain

    record = MatchRecord(
        query_image_path="data/sample_images/refs/ref00.jpg",
        query_image_sha256="ab" * 32,
        post_url="https://www.instagram.com/p/REAL/",
        identified_subject="ref00.jpg",
        platform="Instagram",
        face_verified=True,
    )
    R.anchor_hash(conn, address, record.content_hash())
    assert R.verify_hash(conn, address, record.content_hash()).found

    # Change exactly one hashed field.
    forged = replace(record, post_url="https://www.instagram.com/p/FAKE/")
    assert forged.content_hash() != record.content_hash()
    assert R.verify_hash(conn, address, forged.content_hash()).found is False


def test_volatile_fields_do_not_change_the_anchor(chain):
    """A record re-saved on another machine, at another time, from another
    path must still verify — otherwise the anchor would be useless."""
    conn, address = chain

    record = MatchRecord(
        query_image_path="/home/alice/refs/ref00.jpg",
        query_image_sha256="cd" * 32,
        post_url="https://twitter.com/example/status/1",
        platform="Twitter/X",
        discovered_at="2026-01-01T00:00:00+00:00",
        total_results_found=12,
    )
    R.anchor_hash(conn, address, record.content_hash())

    moved = replace(
        record,
        query_image_path="D:\\other\\machine\\ref00.jpg",
        discovered_at="2026-09-06T23:59:59+00:00",
        total_results_found=99,
        search_engine="google",
    )
    assert moved.content_hash() == record.content_hash()
    assert R.verify_hash(conn, address, moved.content_hash()).found is True


def test_record_file_round_trip_verifies(chain, tmp_path):
    """Save to disk, load back, re-hash — the path verify_onchain.py takes."""
    conn, address = chain
    record = MatchRecord(
        query_image_path="refs/ref01.jpg",
        query_image_sha256="ef" * 32,
        post_url="https://www.linkedin.com/posts/example",
        platform="LinkedIn",
        page_title="Ünïcödé title ✓",
    )
    path = tmp_path / "match_record.json"
    record.save(path)
    R.anchor_hash(conn, address, record.content_hash())

    reloaded = MatchRecord.load(path)
    assert reloaded.content_hash() == record.content_hash()
    assert R.verify_hash(conn, address, reloaded.content_hash()).found is True


def test_anchor_rejects_zero_hash(chain):
    conn, address = chain
    with pytest.raises(R.ChainError):
        R.anchor_hash(conn, address, "0" * 64)


def test_connection_reports_itself_honestly(chain):
    conn, _ = chain
    assert conn.simulated is True
    assert "simulated" in conn.chain_name
    assert "eth-tester" in conn.signing_mode
    assert conn.address in conn.describe()
