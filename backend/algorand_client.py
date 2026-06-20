"""
BlockVerify — Algorand Testnet Interface
=========================================
Handles:
  1. Server wallet creation / loading
  2. Broadcasting model metadata as note-only transactions
  3. Calling the deployed smart contract to store model_id → hash
  4. Live transaction lookup (for Blockchain Proof page)
  5. Wallet account info (for Dashboard)
"""

import json
import os
import time

import requests
from algosdk import account
from algosdk.v2client import algod
from algosdk.transaction import PaymentTxn

# ── Public AlgoNode endpoints (no API key needed) ──────────────────────
ALGOD_ADDRESS     = "https://testnet-api.algonode.cloud"
INDEXER_ADDRESS   = "https://testnet-idx.algonode.cloud"
ALGOD_TOKEN       = ""

client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_ADDRESS)

# ── Friendly error mapping ─────────────────────────────────────────────
_NET_HINTS = (
    "timed out", "timeout", "connection", "max retries", "failed to establish",
    "temporarily unavailable", "service unavailable", "502", "503", "504",
    "name resolution", "getaddrinfo", "unreachable", "remote end closed",
    "connection reset", "connection refused", "ssl",
)

def _friendly_algo_error(exc) -> str:
    """Translate low-level network/SDK failures into a clear, actionable message."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return ("Algorand network unreachable — the Testnet node (AlgoNode) isn't responding "
                "right now. Please try again in a few minutes.")
    low = str(exc).lower()
    if any(h in low for h in _NET_HINTS):
        return ("Algorand network unreachable — the Testnet node (AlgoNode) isn't responding "
                "right now. Please try again in a few minutes.")
    return f"Algorand transaction failed: {exc}"


# ── Wallet persistence ─────────────────────────────────────────────────
WALLET_FILE = os.path.join(os.path.dirname(__file__), "data", "algo_wallet.json")


def get_or_create_wallet():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE) as f:
            data = json.load(f)
            return data["address"], data["private_key"]

    private_key, address = account.generate_account()
    os.makedirs(os.path.dirname(WALLET_FILE), exist_ok=True)
    with open(WALLET_FILE, "w") as f:
        json.dump({"address": address, "private_key": private_key}, f, indent=2)

    print("\n" + "=" * 70)
    print("🚨 NEW ALGORAND TESTNET WALLET GENERATED 🚨")
    print(f"➜  Server Address : {address}")
    print("Fund it (FREE) at: https://bank.testnet.algorand.network/")
    print("=" * 70 + "\n")
    return address, private_key


ADDRESS, PRIVATE_KEY = get_or_create_wallet()


# ── Smart contract auto-deploy / load ─────────────────────────────────

def _get_or_deploy_contract():
    """
    Load the already-deployed App ID from disk, or deploy a fresh contract.
    Called lazily on first registration so the server boots instantly.
    """
    from contract import load_app_id, deploy_contract
    app_id = load_app_id()
    if app_id:
        return app_id

    # Check balance before deploying (costs ~0.001 ALGO)
    try:
        info = client.account_info(ADDRESS)
        if info.get("amount", 0) < 2000:
            print("⚠️  Wallet not funded yet — skipping contract deploy. "
                  f"Fund {ADDRESS} at https://bank.testnet.algorand.network/")
            return None
    except Exception:
        return None

    result = deploy_contract(client, PRIVATE_KEY, ADDRESS)
    return result.get("app_id")


# ── 1. Broadcast hash as note-transaction ─────────────────────────────

def broadcast_hash_to_algorand(model_id, model_name, model_hash, owner):
    """
    Sends a 0-ALGO self-transaction with JSON metadata in the note field.
    Also calls the smart contract to permanently store model_id → hash.
    Returns {"success": True, "txid": ..., "round": ..., "app_id": ..., "contract_txid": ...}
    """
    try:
        account_info = client.account_info(ADDRESS)
        balance = account_info.get("amount", 0)

        if balance < 1000:
            return {
                "success": False,
                "error": (
                    f"Wallet empty! Fund {ADDRESS} at "
                    "https://bank.testnet.algorand.network/"
                ),
            }

        params = client.suggested_params()

        payload = {
            "id":    model_id,
            "name":  model_name,
            "hash":  model_hash,
            "owner": owner,
        }
        note = json.dumps(payload).encode()

        txn = PaymentTxn(
            sender=ADDRESS,
            sp=params,
            receiver=ADDRESS,
            amt=0,
            note=note,
        )
        signed = txn.sign(PRIVATE_KEY)
        txid   = client.send_transaction(signed)

        # Wait for confirmation (~3 s on Algorand)
        confirmed_round = None
        for _ in range(15):
            txinfo = client.pending_transaction_info(txid)
            if txinfo.get("confirmed-round", 0) > 0:
                confirmed_round = txinfo["confirmed-round"]
                break
            time.sleep(1)

        if not confirmed_round:
            return {"success": False, "error": "Tx sent but confirmation timed out."}

        # ── Also call the smart contract ──────────────────────────────
        app_id         = None
        contract_txid  = None
        try:
            from contract import call_contract_register
            app_id = _get_or_deploy_contract()
            if app_id:
                cr = call_contract_register(
                    client, PRIVATE_KEY, ADDRESS,
                    app_id, model_id, model_hash,
                )
                if cr.get("success"):
                    contract_txid = cr.get("contract_txid")
        except Exception as ce:
            # Contract call failure must NOT block registration
            print(f"[contract] Non-fatal: {ce}")

        return {
            "success": True,
            "txid":          txid,
            "round":         confirmed_round,
            "app_id":        app_id,
            "contract_txid": contract_txid,
        }

    except Exception as e:
        return {"success": False, "error": _friendly_algo_error(e)}


# ── 2. Live transaction lookup ─────────────────────────────────────────

def get_transaction_info(txid: str) -> dict:
    """
    Fetch a confirmed transaction by TxID from the Algorand Indexer.
    Returns a cleaned dict suitable for JSON serialisation.
    """
    try:
        url  = f"{INDEXER_ADDRESS}/v2/transactions/{txid}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"success": False, "error": f"Indexer returned {resp.status_code}"}

        data = resp.json().get("transaction", {})

        # Decode note field (base64 → utf-8 JSON if possible)
        note_raw    = data.get("note", "")
        note_decoded = ""
        note_parsed  = None
        if note_raw:
            import base64
            try:
                note_decoded = base64.b64decode(note_raw).decode("utf-8")
                note_parsed  = json.loads(note_decoded)
            except Exception:
                note_decoded = note_raw   # leave as-is

        return {
            "success": True,
            "txid":          data.get("id"),
            "round":         data.get("confirmed-round"),
            "block_time":    data.get("round-time"),
            "sender":        data.get("sender"),
            "receiver":      data.get("payment-transaction", {}).get("receiver"),
            "amount_algo":   data.get("payment-transaction", {}).get("amount", 0) / 1e6,
            "fee_algo":      data.get("fee", 0) / 1e6,
            "note_raw":      note_raw,
            "note_decoded":  note_decoded,
            "note_parsed":   note_parsed,
            "tx_type":       data.get("tx-type"),
            "first_valid":   data.get("first-valid"),
            "last_valid":    data.get("last-valid"),
            "genesis_id":    data.get("genesis-id"),
        }

    except Exception as e:
        return {"success": False, "error": _friendly_algo_error(e)}


# ── 3. Wallet / account info ───────────────────────────────────────────

def get_wallet_info() -> dict:
    """
    Return live account stats for the server wallet from Algorand Testnet.
    Uses direct HTTP with timeout instead of SDK (avoids indefinite blocking).
    """
    try:
        # Direct REST call with explicit timeout
        url  = f"{ALGOD_ADDRESS}/v2/accounts/{ADDRESS}"
        resp = requests.get(url, timeout=8)

        if resp.status_code != 200:
            return {"success": False, "error": f"AlgoNode returned {resp.status_code}"}

        info = resp.json()

        from contract import load_app_id
        app_id       = load_app_id()
        app_explorer = (
            f"https://lora.algokit.io/testnet/application/{app_id}"
            if app_id else None
        )

        return {
            "success":          True,
            "address":          ADDRESS,
            "balance_algo":     info.get("amount", 0) / 1e6,
            "min_balance_algo": info.get("min-balance", 0) / 1e6,
            "status":           "Active (Connected)" if info.get("status") == "Offline" else info.get("status", "Unknown"),
            "created_at_round": info.get("created-at-round"),
            "app_id":           app_id,
            "app_explorer_url": app_explorer,
            "explorer_url":     f"https://lora.algokit.io/testnet/account/{ADDRESS}",
        }

    except requests.Timeout:
        return {"success": False, "error": "AlgoNode timed out (>8s). Check your internet connection."}
    except Exception as e:
        return {"success": False, "error": _friendly_algo_error(e)}


# ── 4. Live Algorand network stats ─────────────────────────────────────

def get_network_stats() -> dict:
    """
    Fetch live Algorand Testnet node status.
    Returns current block round, time-since-last-block, network health.
    """
    try:
        status = client.status()

        last_round  = status.get("last-round", 0)
        # time-since-last-round is in nanoseconds
        ns_since    = status.get("time-since-last-round", 0)
        sec_since   = round(ns_since / 1_000_000_000, 1)

        # Algorand average block time is ~3.3–3.6 seconds on testnet
        avg_block_time = 3.4

        return {
            "success":         True,
            "network":         "Algorand Testnet",
            "last_round":      last_round,
            "sec_since_block": sec_since,
            "avg_block_time":  avg_block_time,
            "is_healthy":      not status.get("stopped-at-unsupported-round", False),
            "node_version":    status.get("last-version", "unknown"),
            "catchup_time":    status.get("catchup-time", 0),
        }

    except Exception as e:
        return {"success": False, "error": _friendly_algo_error(e)}


# ── 5. Transaction history ledger ──────────────────────────────────────

def get_transaction_ledger(limit: int = 50) -> dict:
    """
    Fetch recent transactions from the server wallet via the Algorand Indexer.
    Parses the note field to extract model metadata from BlockVerify txns.
    """
    try:
        url = (
            f"{INDEXER_ADDRESS}/v2/accounts/{ADDRESS}/transactions"
            f"?limit={limit}"
        )
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return {"success": False, "error": f"Indexer returned {resp.status_code}"}

        data = resp.json()
        raw_txns = data.get("transactions", [])

        ledger = []
        for tx in raw_txns:
            entry = {
                "txid":        tx.get("id", ""),
                "round":       tx.get("confirmed-round", 0),
                "timestamp":   tx.get("round-time", 0),
                "type":        tx.get("tx-type", ""),
                "fee":         tx.get("fee", 0) / 1e6,
                "explorer_url": f"https://lora.algokit.io/testnet/transaction/{tx.get('id', '')}",
            }

            # Try to decode the note field for BlockVerify metadata
            import base64
            raw_note = tx.get("note", "")
            if raw_note:
                try:
                    decoded = base64.b64decode(raw_note).decode("utf-8")
                    meta = json.loads(decoded)
                    entry["model_name"] = meta.get("name", "")
                    entry["model_id"]   = meta.get("id", "")
                    entry["model_hash"] = meta.get("hash", "")[:16] + "..."
                    entry["owner"]      = meta.get("owner", "")
                    entry["action"]     = "registration"
                except (json.JSONDecodeError, UnicodeDecodeError):
                    entry["action"] = "contract-call"
            else:
                # Application call (smart contract) or other tx
                if tx.get("tx-type") == "appl":
                    entry["action"] = "smart-contract"
                else:
                    entry["action"] = "transfer"

            ledger.append(entry)

        return {
            "success":    True,
            "count":      len(ledger),
            "ledger":     ledger,
            "wallet":     ADDRESS,
        }

    except requests.Timeout:
        return {"success": False, "error": "Indexer timed out (>12s)"}
    except Exception as e:
        return {"success": False, "error": _friendly_algo_error(e)}
