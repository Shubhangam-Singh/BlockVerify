# Deploy BlockVerify on Oracle Cloud (Always Free)

One small Linux VM runs everything: the website, the backend API and the live-alerts socket,
under one HTTPS address. Unlike Render's free tier it never sleeps, and its disk keeps your
accounts, models, photos and login keys (`backend/data`) across restarts.

```
browser ──HTTPS──> Caddy (ports 80/443, automatic certificate)
                     ├── /api/*, /socket.io/*  ──> BlockVerify backend (127.0.0.1:5000, systemd)
                     └── everything else       ──> /var/www/blockverify (the frontend)
```

Time needed: about an hour, most of it in the Oracle console.

---

## 1. Create the Oracle Cloud account

1. Sign up at <https://signup.cloud.oracle.com>. Oracle asks for a card to verify you; Always Free
   resources are not charged.
2. Pick your **home region** carefully, because it cannot be changed later. From India, choose
   *India West (Mumbai)* or *India South (Hyderabad)*.

## 2. Create the VM

Console → **Compute → Instances → Create instance**.

| Setting | Choose |
|---|---|
| Name | `blockverify` |
| Image | **Canonical Ubuntu 24.04** (Change image → Ubuntu) |
| Shape | Change shape → **Ampere → VM.Standard.A1.Flex**, 1 OCPU, 6 GB memory |
| Networking | Create a new virtual cloud network and public subnet; **Assign a public IPv4 address** |
| SSH keys | **Generate a key pair** and download the private key (or paste your own public key) |

Click **Create** and wait until it says *Running*. Copy the **Public IP address**.

> If you get *"Out of capacity"* for the Ampere shape, try another availability domain, try again
> later, or use the AMD shape **VM.Standard.E2.1.Micro** (also Always Free; 1 GB RAM is enough).

## 3. Open ports 80 and 443 in Oracle's network

Console → **Networking → Virtual cloud networks →** your VCN **→ Security → Default Security List
→ Add Ingress Rules**. Add two rules:

| Source CIDR | IP protocol | Destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

(The setup script opens the same ports in the VM's own firewall.)

## 4. Get a free domain name

HTTPS needs a name, not a bare IP. The free option:

1. Go to <https://www.duckdns.org> and sign in (GitHub or Google).
2. Add a subdomain, e.g. `blockverify` → you get `blockverify.duckdns.org`.
3. Put the VM's public IP in **current ip** and click **update ip**.

(If you own a domain, add an `A` record pointing to the VM's IP instead.)

## 5. Log in to the VM

On your laptop (macOS, Linux, or Windows PowerShell):

```bash
chmod 600 ~/Downloads/ssh-key-*.key
```

```bash
ssh -i ~/Downloads/ssh-key-*.key ubuntu@<PUBLIC-IP>
```

On Windows, skip `chmod` and use the full path to the key, e.g. `ssh -i C:\Users\you\Downloads\ssh-key-2026-11-03.key ubuntu@<PUBLIC-IP>`.

## 6. Run the setup script

On the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
```

```bash
git clone --branch ss2005/modest-fermat-hflkcl https://github.com/Shubhangam-Singh/BlockVerify.git ~/bv-setup && bash ~/bv-setup/deploy/oracle/setup.sh blockverify.duckdns.org
```

Use your own DuckDNS name. Once the pull request is merged, you can pass `main` as a second
argument to deploy from `main` instead. The script installs Python and Caddy, puts the app in
`/opt/blockverify`, creates the `blockverify` service and the Caddy config, and opens the
firewall. It is safe to run again.

## 7. Add the Algorand wallet and start the backend

The backend is not started until it has a wallet, so it never makes a throwaway one.

```bash
sudo nano /etc/blockverify.env
```

Set `ALGO_PRIVATE_KEY=` to the `private_key` value from your laptop's
`backend/data/algo_wallet.json`. Save with Ctrl+O, Enter, Ctrl+X. Then:

```bash
sudo systemctl restart blockverify
```

```bash
sudo journalctl -u blockverify -n 30 --no-pager
```

The log should show the BlockVerify banner and **no** "NEW ALGORAND TESTNET WALLET GENERATED" line.

## 8. Open the site

Go to `https://blockverify.duckdns.org`. The first visit can take up to a minute while Caddy gets
the HTTPS certificate. Create an account, then test with the files in `test_models/`.

---

## Everyday commands (on the VM)

| What | Command |
|---|---|
| Update to the latest code | `bash /opt/blockverify/deploy/oracle/update.sh` |
| Backend logs (live) | `sudo journalctl -u blockverify -f` |
| Restart the backend | `sudo systemctl restart blockverify` |
| Web server logs | `sudo journalctl -u caddy -n 50 --no-pager` |
| Edit settings | `sudo nano /etc/blockverify.env`, then restart the backend |

`update.sh` pulls the branch, reinstalls requirements, republishes the frontend and restarts the
backend. It never touches `backend/data`.

## Back up your data

Everything users create lives in `/opt/blockverify/backend/data`. On the VM:

```bash
tar czf ~/blockverify-data-$(date +%F).tgz -C /opt/blockverify/backend data
```

Copy it to your laptop (run this on the laptop):

```bash
scp -i ~/Downloads/ssh-key-*.key ubuntu@<PUBLIC-IP>:~/blockverify-data-*.tgz .
```

The archive contains the wallet key and the login secret, so keep it private.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Browser can't reach the site | Check the two ingress rules (step 3) and that DuckDNS shows the VM's current IP. |
| Certificate error | DNS must point at the VM before Caddy can get a certificate. Fix DuckDNS, then `sudo systemctl restart caddy`. |
| Site loads but says the server is unreachable | `sudo systemctl status blockverify`; if it is not running, check the wallet line in `/etc/blockverify.env` and the logs. |
| Registration fails with "Wallet empty" | Fund the wallet address at <https://bank.testnet.algorand.network/>. |
| The VM stopped by itself | Oracle may reclaim Always Free VMs it considers idle. Upgrading the account to Pay As You Go (still free inside the limits) is the usual way to avoid that. |

## How the frontend finds the backend

The repository's `frontend/index.html` calls the Render backend. `publish-frontend.sh` copies the
frontend to `/var/www/blockverify` and changes that one line in the copy so it calls
`https://<your domain>/api`. The repository file is not modified, so Render keeps working.
