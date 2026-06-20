"""
bitcoin_client.py
=================
Bitcoin Testnet OP_RETURN anchor module for BlockVerify.

Architecture:
  - Algorand Testnet  → fast sidechain (individual model hashes)
  - Bitcoin Testnet   → L1 settlement layer (Merkle Root of all models)

This module takes the Merkle Root hash, constructs an OP_RETURN transaction,
and broadcasts it to Bitcoin Testnet via the Blockstream API.

Author: razor_hats team | Branch: future-bitcoin-integration
"""

import hashlib
import os
import requests
try:
    from bit import PrivateKeyTestnet
    BIT_AVAILABLE = True
except ImportError:
    BIT_AVAILABLE = False
    PrivateKeyTestnet = None

# ── Wallet configuration ────────────────────────────────────────────────────
# Bitcoin *Testnet* WIF — override via the BTC_WIF environment variable (.env).
# The fallback below is a throwaway testnet key (no real-money value) kept only so
# the demo works out-of-the-box; set BTC_WIF in your .env to use your own wallet.
# Address: mrDTrvKrLpW969E8CbqagN8KRRJ3u49huZ
BTC_WIF = os.environ.get("BTC_WIF", "cVTqu1VtqxqzAkKNAQUTVh5FFwcbCRPEeo4EaH5TzgLw8UQpCU7y")

BLOCKSTREAM_API   = "https://blockstream.info/testnet/api"
EXPLORER_BASE_URL = "https://blockstream.info/testnet/tx"

# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_key():
    if not BIT_AVAILABLE:
        raise RuntimeError("Bitcoin library (bit) is not installed on this server.")
    return PrivateKeyTestnet(BTC_WIF)


def get_wallet_balance() -> dict:
    """Return the current Testnet wallet balance in satoshis and BTC."""
    if not BIT_AVAILABLE:
        return {"success": False, "error": "Bitcoin library not available on this server. Run locally to use this feature."}
    try:
        key = _get_key()
        sat = int(key.get_balance('satoshi'))
        return {
            "success":  True,
            "address":  key.address,
            "satoshi":  sat,
            "btc":      sat / 1e8,
            "explorer": f"https://blockstream.info/testnet/address/{key.address}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_op_return_script_hex(merkle_root: str) -> str:
    """
    Build the OP_RETURN output script for a 32-byte (SHA-256) payload.
    Script: OP_RETURN (0x6a) | push 32 bytes (0x20) | <32-byte-hash>
    """
    data_bytes = bytes.fromhex(merkle_root[:64])  # take first 32 bytes (64 hex chars)
    assert len(data_bytes) == 32, "Merkle root must be 32 bytes"
    return "6a20" + data_bytes.hex()


def anchor_merkle_root(merkle_root: str) -> dict:
    """
    Broadcast an OP_RETURN transaction embedding the Merkle Root to Bitcoin Testnet.

    Returns a dict with:
      success, txid, explorer_url, op_return_hex, fee_satoshi, balance_before
    """
    try:
        if not merkle_root or len(merkle_root) < 64:
            return {"success": False, "error": "Invalid Merkle Root — must be a 64-char hex string."}

        key = _get_key()

        # Check we have UTXOs to spend
        utxos = key.get_unspents()
        if not utxos:
            return {
                "success": False,
                "error": "Wallet has no UTXOs. Send some Testnet BTC to " + key.address,
            }

        balance_sat = sum(u.amount for u in utxos)

        # Build OP_RETURN script metadata
        op_return_hex = build_op_return_script_hex(merkle_root)

        # The 'bit' library supports a plain string message (max 80 bytes after utf-8 encoding).
        # We embed the 64-char hex string of the merkle root (64 bytes, within OP_RETURN limit).
        # bit adds OP_RETURN automatically.
        tx_hex = key.create_transaction(
            outputs=[],
            fee=2000,               # 2000 satoshi absolute fee (~2 sat/byte on testnet)
            absolute_fee=True,      # interpret fee as exact satoshi amount
            leftover=key.address,   # send change back to ourselves
            combine=True,
            message=merkle_root[:64],  # 64-char ASCII hex string → 64 bytes
            unspents=utxos,
        )

        # Broadcast via Blockstream Testnet API
        resp = requests.post(
            f"{BLOCKSTREAM_API}/tx",
            data=tx_hex,
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )

        if resp.status_code != 200:
            return {
                "success":       False,
                "error":         f"Blockstream rejected transaction: {resp.text[:200]}",
                "tx_hex":        tx_hex,
                "op_return_hex": op_return_hex,
            }

        txid = resp.text.strip()

        return {
            "success":            True,
            "txid":               txid,
            "explorer_url":       f"{EXPLORER_BASE_URL}/{txid}",
            "op_return_hex":      op_return_hex,
            "op_return_data":     merkle_root[:64],
            "fee_satoshi":        2000,
            "balance_before_sat": balance_sat,
            "wallet_address":     key.address,
            "network":            "Bitcoin Testnet3",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def decode_op_return_annotated(op_return_hex: str) -> list[dict]:
    """
    Return a list of annotated byte-groups for educational display.
    Input: '6a20<32-bytes>'
    """
    return [
        {"bytes": "6a",          "label": "OP_RETURN",   "description": "Marks output as provably unspendable — used for embedding data"},
        {"bytes": "20",          "label": "PUSH 32",     "description": "Push the next 32 bytes onto the stack"},
        {"bytes": op_return_hex[4:], "label": "Merkle Root", "description": "SHA-256 Merkle Root of all BlockVerify model hashes"},
    ]
