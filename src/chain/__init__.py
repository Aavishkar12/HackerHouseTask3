"""Stage 3: anchoring Stage 2 match records on a blockchain."""

from .registry import (  # noqa: F401
    AnchorResult,
    ChainConnection,
    ChainError,
    VerifyResult,
    anchor_hash,
    connect,
    deploy,
    load_artifact,
    load_deployment,
    normalize_hash,
    resolve_contract_address,
    save_deployment,
    total_anchored,
    verify_hash,
)
