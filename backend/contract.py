"""
BlockVerify — Algorand Smart Contract (TEAL v8)
================================================
A simple model registry contract deployed on Algorand Testnet.
Global state stores: model_id (bytes) → model_hash (bytes)

Deploy once; call "register" for every model registration.
"""

import base64
import json
import os
import time

from algosdk import account
from algosdk.v2client import algod
from algosdk.transaction import (
    ApplicationCreateTxn,
    ApplicationNoOpTxn,
    StateSchema,
    wait_for_confirmation,
)

# ──────────────────────────────────────────────
#  TEAL source programs
# ──────────────────────────────────────────────

_CONTRACT_DIR = os.path.dirname(__file__)

def _load_teal(filename: str) -> str:
    """Load TEAL source from a .teal file next to this module."""
    path = os.path.join(_CONTRACT_DIR, filename)
    with open(path) as f:
        return f.read()

# Read TEAL programs from their dedicated .teal files
# (These files are also tracked by GitHub Linguist as 'TEAL' language)
APPROVAL_PROGRAM = _load_teal("approval.teal")
CLEAR_PROGRAM    = _load_teal("clear.teal")

# Supports up to 64 registered models per contract deployment
GLOBAL_SCHEMA = StateSchema(num_uints=0, num_byte_slices=64)
LOCAL_SCHEMA  = StateSchema(num_uints=0, num_byte_slices=0)

APP_ID_FILE = os.path.join(os.path.dirname(__file__), "data", "algo_app.json")


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _compile(algod_client: algod.AlgodClient, source: str) -> bytes:
    resp = algod_client.compile(source)
    return base64.b64decode(resp["result"])


def save_app_id(app_id: int, txid: str) -> None:
    os.makedirs(os.path.dirname(APP_ID_FILE), exist_ok=True)
    with open(APP_ID_FILE, "w") as f:
        json.dump({"app_id": app_id, "deploy_txid": txid}, f, indent=2)


def load_app_id() -> int | None:
    # ALGO_APP_ID lets a host with an ephemeral disk reuse the deployed contract
    # instead of deploying a new one after every restart.
    env_id = (os.environ.get("ALGO_APP_ID") or "").strip()
    if env_id.isdigit():
        return int(env_id)
    if os.path.exists(APP_ID_FILE):
        with open(APP_ID_FILE) as f:
            return json.load(f).get("app_id")
    return None


# ──────────────────────────────────────────────
#  Deploy
# ──────────────────────────────────────────────

def deploy_contract(algod_client: algod.AlgodClient,
                    private_key: str, sender: str) -> dict:
    """
    Compile and deploy the approval + clear programs.
    Returns {"success": True, "app_id": int, "txid": str}
    """
    try:
        approval_bytes = _compile(algod_client, APPROVAL_PROGRAM)
        clear_bytes    = _compile(algod_client, CLEAR_PROGRAM)

        sp = algod_client.suggested_params()

        txn = ApplicationCreateTxn(
            sender=sender,
            sp=sp,
            on_complete=0,          # NoOpOC
            approval_program=approval_bytes,
            clear_program=clear_bytes,
            global_schema=GLOBAL_SCHEMA,
            local_schema=LOCAL_SCHEMA,
        )

        signed = txn.sign(private_key)
        txid   = algod_client.send_transaction(signed)

        result  = wait_for_confirmation(algod_client, txid, 10)
        app_id  = result["application-index"]

        save_app_id(app_id, txid)

        print(f"\n✅ BlockVerify Smart Contract DEPLOYED on Algorand Testnet!")
        print(f"   App ID  : {app_id}")
        print(f"   Deploy TxID : {txid}")
        print(f"   Explorer: https://lora.algokit.io/testnet/application/{app_id}\n")

        return {"success": True, "app_id": app_id, "txid": txid}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────
#  Call: register model on-chain
# ──────────────────────────────────────────────

def call_contract_register(algod_client: algod.AlgodClient,
                            private_key: str, sender: str,
                            app_id: int,
                            model_id: str, model_hash: str) -> dict:
    """
    Call the 'register' method on the deployed smart contract.
    Stores model_id → model_hash in the contract's global state.
    """
    try:
        sp = algod_client.suggested_params()

        txn = ApplicationNoOpTxn(
            sender=sender,
            sp=sp,
            index=app_id,
            app_args=[
                b"register",
                model_id[:64].encode(),    # key  (max 64 bytes per Algorand limit)
                model_hash[:64].encode(),  # value
            ],
        )

        signed = txn.sign(private_key)
        txid   = algod_client.send_transaction(signed)

        result = wait_for_confirmation(algod_client, txid, 10)

        return {
            "success": True,
            "contract_txid": txid,
            "confirmed_round": result.get("confirmed-round"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────
#  Read: verify model hash from on-chain state
# ──────────────────────────────────────────────

def read_contract_state(app_id: int, model_id: str) -> dict:
    """
    Read the smart contract's global state for a specific model_id.
    Returns the on-chain hash so the caller can compare it with the local hash.
    Uses direct HTTP to avoid SDK timeout issues.
    """
    import requests

    try:
        algod_url = "https://testnet-api.algonode.cloud"
        url = f"{algod_url}/v2/applications/{app_id}"
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            return {"success": False, "error": f"AlgoNode returned {resp.status_code}"}

        app_data = resp.json()
        global_state = (
            app_data.get("params", {})
            .get("global-state", [])
        )

        # Algorand stores keys as base64-encoded bytes
        target_key_b64 = base64.b64encode(
            model_id[:64].encode()
        ).decode()

        for entry in global_state:
            if entry.get("key") == target_key_b64:
                # Value is a byte-slice (type 1)
                value = entry.get("value", {})
                if value.get("type") == 1:
                    on_chain_hash = base64.b64decode(
                        value.get("bytes", "")
                    ).decode("utf-8", errors="replace")
                    return {
                        "success":       True,
                        "found":         True,
                        "model_id":      model_id,
                        "on_chain_hash": on_chain_hash,
                        "app_id":        app_id,
                    }

        # Key not found in global state
        return {
            "success":  True,
            "found":    False,
            "model_id": model_id,
            "app_id":   app_id,
            "message":  "Model ID not found in smart contract global state",
        }

    except requests.Timeout:
        return {"success": False, "error": "AlgoNode timed out (>10s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}
