// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title MatchRegistry
 * @notice Anchors face-match discoveries from Stage 2 on-chain.
 *
 * Design notes
 * ------------
 * The chain deliberately stores NO personal data — no image, no URL, no
 * name. It stores only a 32-byte fingerprint of the Stage 2 match record
 * (a SHA-256 over a canonical JSON payload), plus when it was anchored
 * and by whom.
 *
 * That is enough to prove, later, that a given match record existed in
 * exactly its current form at the time it was anchored:
 *
 *     re-hash the record  ->  same bytes32  ->  found on chain  ->  verified
 *     edit ANY hashed field -> different bytes32 -> not found   -> tampered
 *
 * Anchoring is idempotent-safe: re-anchoring the same hash reverts rather
 * than silently overwriting the original timestamp, so the first anchor
 * time is immutable.
 */
contract MatchRegistry {
    struct Record {
        uint256 timestamp; // block time of the anchoring transaction
        address submitter; // who anchored it
        bool exists; // distinguishes "anchored" from "never seen"
    }

    mapping(bytes32 => Record) private _records;
    bytes32[] private _anchored;

    event MatchAnchored(
        bytes32 indexed contentHash,
        address indexed submitter,
        uint256 timestamp
    );

    error AlreadyAnchored(bytes32 contentHash, uint256 timestamp);
    error EmptyHash();

    /// @notice Anchor a Stage 2 match-record fingerprint.
    function anchor(bytes32 contentHash) external {
        if (contentHash == bytes32(0)) revert EmptyHash();

        Record storage existing = _records[contentHash];
        if (existing.exists) {
            revert AlreadyAnchored(contentHash, existing.timestamp);
        }

        _records[contentHash] = Record({
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });
        _anchored.push(contentHash);

        emit MatchAnchored(contentHash, msg.sender, block.timestamp);
    }

    /// @notice Look up a fingerprint. This is the verification call.
    /// @return exists    true if this exact record was ever anchored
    /// @return timestamp block time it was anchored (0 if never)
    /// @return submitter address that anchored it (zero address if never)
    function verify(bytes32 contentHash)
        external
        view
        returns (bool exists, uint256 timestamp, address submitter)
    {
        Record storage r = _records[contentHash];
        return (r.exists, r.timestamp, r.submitter);
    }

    /// @notice How many distinct records have been anchored.
    function total() external view returns (uint256) {
        return _anchored.length;
    }

    /// @notice Enumerate anchored fingerprints (for demo/audit output).
    function hashAt(uint256 index) external view returns (bytes32) {
        return _anchored[index];
    }
}
