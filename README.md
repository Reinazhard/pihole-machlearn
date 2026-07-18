# Pi-hole ML Ad/Tracker Detector

**Zero-latency, on-device deep learning DNS sinkhole.** Classifies domains in real-time using a character-level Convolutional Neural Network (Char-CNN) exported to ONNX, blocking ads and trackers before they resolve. No cloud APIs, no external services, no added latency.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-green.svg)](https://onnxruntime.ai/)
[![Pi-hole 5.x/6.x](https://img.shields.io/badge/Pi--hole-5.x%2F6.x-red.svg)](https://pi-hole.net/)

---

## Table of Contents

- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Deep Learning Model](#deep-learning-model)
- [Fail-Safes & False Positive Mitigation](#fail-safes--false-positive-mitigation)
- [DGA Wildcard Aggregation](#dga-wildcard-aggregation)
- [Prerequisites & Dependencies](#prerequisites--dependencies)
- [Deployment & Installation](#deployment--installation)
- [Configuration Reference](#configuration-reference)
- [Usage & Verification](#usage--verification)
- [Edge Cases & Troubleshooting](#edge-cases--troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

---

## System Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │            ml-detector (Docker)             │
                          │                                             │
                          │   ┌──────────┐  ┌─────────┐  ┌──────────┐  │
                          │   │detector.py│  │sweep.py │  │train.py  │  │
                          │   │ */5 min   │  │ 3:00 AM │  │ 1st 2 AM │  │
                          │   └────┬─────┘  └────┬────┘  └────┬─────┘  │
                          │        │              │             │        │
                          │        ▼              ▼             ▼        │
  ┌───────────┐           │   ┌─────────────────────────────────────┐   │
  │ Pi-hole   │◄──────────┼──►│          SQLite Databases           │   │
  │ FTL DNS   │           │   │  gravity.db  ·  pihole-FTL.db      │   │
  │ Resolver  │           │   └──────────────┬──────────────────────┘   │
  └─────┬─────┘           │                  │                          │
        │                 │                  ▼                          │
        │                 │   ┌─────────────────────────────────────┐   │
        │                 │   │       Char-CNN (ONNX Runtime)       │   │
        │                 │   │  256-byte vocab → 32-dim embed      │   │
        │                 │   │  3× Conv1D(128) → FC(128→2)         │   │
        │                 │   │  Output: [Safe, Ad/Tracker]         │   │
        │                 │   └──────────────┬──────────────────────┘   │
        │                 │                  │                          │
        │                 │                  ▼                          │
        │                 │   ┌─────────────────────────────────────┐   │
        │                 │   │     Pi-hole domainlist (SQLite)     │   │
        │                 │   │  type 1 = exact block               │   │
        │                 │   │  type 3 = regex wildcard            │   │
        │                 │   └──────────────┬──────────────────────┘   │
        │                 │                  │                          │
        │                 │                  ▼                          │
        │   reloaddns     │   ┌─────────────────────────────────────┐   │
        └─────────────────┼──►│    docker exec pihole pihole -g      │   │
                          │   └─────────────────────────────────────┘   │
                          └─────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion** — `detector.py` queries `pihole-FTL.db` (read-only) for all domains resolved in the last 5 minutes with status `forwarded` (2) or `cached` (3).
2. **Deduplication** — Domains already present in `gravity.db`'s `domainlist` table are excluded to avoid re-evaluation.
3. **Bypass Filtering** — The Majestic Top 100k allowlist and CDN infrastructure suffix rules remove known-safe domains before inference.
4. **Inference** — Remaining domains are encoded as byte sequences (truncated/padded to 100 bytes) and classified by the Char-CNN ONNX model.
5. **Blocking** — Domains with `P(ad) > 0.95` are inserted into `domainlist` as exact blocks (type 1) with the comment `Added by ML Detector (Char-CNN High Confidence)`.
6. **DGA Aggregation** — If 3+ subdomains of a root domain are blocked, a regex wildcard (type 3) is inserted to catch future subdomains automatically.
7. **DNS Reload** — `pihole reloaddns` is invoked via the Docker socket to apply changes immediately.

---

## How It Works

This system is **not** an LLM-based API call. Despite the repository name, it deploys a lightweight character-level Convolutional Neural Network (Char-CNN) that runs entirely on-device via ONNX Runtime. The model processes raw domain strings byte-by-byte, learning structural patterns that distinguish advertising and tracking domains from legitimate ones without any feature engineering, DNS lookups, or external API calls.

The pipeline runs as three scheduled cron jobs inside a Docker container:

| Component | Schedule | Purpose |
|-----------|----------|---------|
| `detector.py` | Every 5 minutes | Classifies recently resolved domains and blocks ads/trackers |
| `sweep.py` | Daily at 3:00 AM | Consolidates ML-added exact blocks into a single adlist file |
| `train.py` | Monthly on the 1st at 2:00 AM | Retrains the model on fresh ad/safe domain data |

---

## Deep Learning Model

### Architecture: Char-CNN

The model processes domain names as raw byte sequences, eliminating the need for manual feature engineering.

| Layer | Configuration |
|-------|---------------|
| Embedding | 256-dim byte vocabulary → 32-dim dense vectors |
| Conv1D Block ×3 | 128 filters, kernel size 3, padding 1, ReLU activation |
| Pooling | MaxPool1D (factor 2) after each conv block |
| Fully Connected 1 | 128 × (100 ÷ 8) = 1,536 → 128, ReLU |
| Dropout | Rate 0.5 |
| Fully Connected 2 | 128 → 2 (output classes) |
| Output | Softmax probabilities for `[Safe, Ad/Tracker]` |

### Input Encoding

- Each domain is UTF-8 encoded and truncated or zero-padded to exactly **100 bytes**
- Byte values (0–255) are used directly as vocabulary indices
- No tokenization, no stemming, no feature extraction

### Training Details

| Parameter | Value |
|-----------|-------|
| Positive samples | 150,000 ad domains from `gravity.db` |
| Negative samples | 150,000 safe domains from the Majestic Million |
| Train/test split | 80% / 20% (shuffled) |
| Loss function | Weighted CrossEntropyLoss |
| Class weights | `[3.0, 1.0]` — false positives penalized 3× harder |
| Optimizer | Adam (lr=0.001) |
| Epochs | 5 |
| Batch size | 512 |
| Export format | ONNX (opset 18), CPU inference via `onnxruntime` |

The class weighting is a deliberate design choice: the model is trained to err on the side of letting ads through rather than blocking legitimate domains. This is complemented by the 0.95 confidence threshold applied at inference time.

### Alternatives Considered

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| TF-IDF + Logistic Regression | Fast, interpretable | Brittle to new TLDs, misses byte patterns | Deprecated |
| Char-CNN (current) | Captures subdomain structure, no feature engineering | Requires ONNX runtime (~50 MB) | Active |
| Transformer-based | Better contextual understanding | 100ms+ latency per domain, heavy memory | Overkill |

---

## Fail-Safes & False Positive Mitigation

False positives in a DNS sinkhole break real services. This system implements three defense layers, applied before inference.

### 1. Majestic Top 100k Bypass

A read-only snapshot of the [Majestic Million](https://downloads.majestic.com/majestic_million.csv) top domains acts as a strict allowlist. Any domain appearing in the top 100k by traffic rank is **never** classified, regardless of model output. The snapshot is refreshed on each monthly retrain cycle.

### 2. CDN / Infrastructure Suffix Bypass

Known CDN and cloud infrastructure domains are bypassed via suffix matching. These are trust boundaries — blocking them breaks legitimate services.

| Suffix | Provider |
|--------|----------|
| `.googleapis.com` | Google Cloud |
| `.akamaihd.net` | Akamai CDN |
| `.cloudfront.net` | AWS CloudFront |
| `.amazonaws.com` | AWS |
| `.shopeemobile.com` | Shopee CDN |
| `.fbcdn.net` | Meta/Facebook CDN |
| `.googleusercontent.com` | Google |
| `.susercontent.com` | Shopee |
| `.gstatic.com` | Google Static |
| `.whatsapp.net` | WhatsApp/Meta |

The suffix list is intentionally conservative and manually curated. New entries should be added only after verifying they are infrastructure domains, not ad-serving subdomains.

### 3. High Confidence Threshold

Only domains with `P(ad) > 0.95` are blocked. This is deliberately high to minimize false positives. The trade-off is that some borderline ad domains may pass through unblocked.

---

## DGA Wildcard Aggregation

When the detector identifies multiple suspicious subdomains under the same root domain, it automatically upgrades from exact blocking to regex wildcard blocking. This catches both current and future subdomains without requiring individual classification.

### Trigger Logic

```
If 3+ subdomains of example.com are in the blocklist (type 1):
  → Insert regex: (\.|^)example\.com$
  → All current and future subdomains are blocked automatically
```

### Example

```
ad1.tracking.com → blocked (exact)
ad2.tracking.com → blocked (exact)
ad3.tracking.com → triggers aggregation → (\.|^)tracking\.com$ (wildcard)
ad4.tracking.com → caught by wildcard automatically (no inference needed)
```

The wildcard regex is inserted into Pi-hole's `domainlist` table as type 3 (regex blacklist). Individual exact entries for aggregated root domains are cleaned up by the daily sweep job.

---

## Prerequisites & Dependencies

### Minimum Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| OS | Linux (amd64/arm64) | Container host |
| Docker | 20.10+ | Container runtime |
| Docker Compose | v2+ | Service orchestration |
| Pi-hole | 5.x or 6.x | DNS sinkhole (running in Docker) |
| Python | 3.10+ | Only needed for bare-metal deployment |
| Disk | ~500 MB | Container image + model files |
| RAM | 2 GB minimum (8 GB allocated) | Training requires more; inference is lightweight |

### Pi-hole Docker Requirements

The Pi-hole container must be:

- Named `pihole` (or adjust the `docker exec` commands in the scripts)
- Connected to a Docker network named `dns`
- Sharing its `/etc/pihole` directory with the ml-detector container

### Python Dependencies

Declared in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrame operations for domain processing |
| `tldextract` | Domain parsing for DGA wildcard aggregation |
| `torch` | Char-CNN model definition and training |
| `onnx` | Model export to ONNX format |
| `onnxruntime` | CPU inference engine |
| `onnxscript` | ONNX export support |
| `scikit-learn` | Training utilities |
| `joblib` | Serialization support |

---

## Deployment & Installation

### Option 1: Containerized (Recommended)

This is the primary deployment method. The ml-detector runs as a Docker container with cron-based scheduling.

#### Step 1: Train the Model

Training must run once before the first deployment. This produces the `model.onnx` file.

```bash
cd pihole-llm

# Create virtual environment (if not already present)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Train the model (2-5 minutes on CPU)
./venv/bin/python train.py
```

Output: `model.onnx` in the project root.

#### Step 2: Prepare the Network

The ml-detector container must join the same Docker network as your Pi-hole container.

```bash
# Find your Pi-hole's network
docker inspect pihole --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'

# If it's not named 'dns', create a shared network and connect both containers
docker network create dns
docker network connect dns pihole
```

#### Step 3: Configure Volumes

Edit `docker-compose.yml` and adjust the volume mapping for your Pi-hole data directory. The default assumes `./pihole/etc-pihole` on the host maps to `/etc/pihole` in the container:

```yaml
volumes:
  - ./pihole/etc-pihole:/etc/pihole          # Adjust host path to match your Pi-hole installation
  - /var/run/docker.sock:/var/run/docker.sock  # Required for pihole reloaddns
  - ./model.onnx:/app/model.onnx              # Trained model
```

#### Step 4: Build and Deploy

```bash
docker compose up -d --build ml-detector
```

#### Step 5: Verify

```bash
docker logs -f ml-detector
```

You should see the detector processing domains every 5 minutes.

---

### Option 2: Bare-Metal (Cron-based)

For environments where Docker is not available or Pi-hole runs natively.

#### Step 1: Set Up the Environment

```bash
cd pihole-llm
chmod +x setup.sh
./setup.sh
```

This creates a Python virtual environment in `./venv` and installs all dependencies.

#### Step 2: Train the Model

```bash
./venv/bin/python train.py
```

#### Step 3: Run the Detector Manually (Test)

```bash
./venv/bin/python detector.py
```

Verify output shows domains being processed and classified.

#### Step 4: Install Cron Jobs

```bash
# Add cron jobs: detector every 5 min, sweep daily at 3 AM, train monthly
(crontab -l 2>/dev/null; cat <<EOF
*/5 * * * * $(pwd)/venv/bin/python $(pwd)/detector.py >> $(pwd)/detector.log 2>&1
0 3 * * * $(pwd)/venv/bin/python $(pwd)/sweep.py >> $(pwd)/sweep.log 2>&1
0 2 1 * * $(pwd)/venv/bin/python $(pwd)/train.py >> $(pwd)/train.log 2>&1
EOF
) | crontab -
```

#### Step 5: Verify Cron Installation

```bash
crontab -l
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAVITY_DB` | `/etc/pihole/gravity.db` | Path to Pi-hole's gravity database. Read by all three scripts. Used for domain list lookups, block insertion, and training data extraction. |
| `FTL_DB` | `/etc/pihole/pihole-FTL.db` | Path to Pi-hole's FTL query database. Read-only by `detector.py` to fetch recently resolved domains. |
| `ML_LIST_FILE` | `/etc/pihole/ml-blocklist.txt` | Path to the consolidated ML blocklist file. Written by `sweep.py` and registered as a Pi-hole adlist source. |

### Hardcoded Constants

These are defined at the top of each script and can be modified before deployment:

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `TIME_WINDOW_SEC` | `detector.py` | `300` (5 min) | Lookback window for recent queries from `pihole-FTL.db` |
| `MAX_LEN` | `detector.py`, `train.py` | `100` | Maximum domain byte length for model input |
| `CONFIDENCE_THRESHOLD` | `detector.py` | `0.95` | Minimum `P(ad)` probability to trigger blocking |
| `BATCH_SIZE` | `train.py` | `512` | Training batch size |
| `COMMENT_FLAG` | `sweep.py` | `Added by ML Detector` | Comment substring used to identify ML-added entries in `domainlist` |

### Docker Resource Limits

Defined in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 8G
```

Training requires significant CPU and memory. Inference (detector.py) is lightweight and can run with lower limits. Adjust based on your host hardware.

### Logging

Container logs are managed by the `json-file` driver with rotation:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Usage & Verification

### Confirm the Detector Is Running

```bash
# Containerized
docker logs ml-detector --tail 50

# Bare-metal
tail -50 detector.log
```

Expected output pattern:

```
Fetching recent queries...
Checking 47 domains against existing blocklists...
Loading Top 10k Safe Domains bypass list...
Loading ONNX model and predicting...
No new ad/tracker domains detected in this batch exceeding the confidence threshold.
```

### Manually Trigger the Detector

```bash
# Containerized
docker exec ml-detector python /app/detector.py

# Bare-metal
./venv/bin/python detector.py
```

### Manually Trigger the Sweep

```bash
# Containerized
docker exec ml-detector python /app/sweep.py

# Bare-metal
./venv/bin/python sweep.py
```

### Manually Trigger a Retrain

```bash
# Containerized
docker exec ml-detector python /app/train.py

# Bare-metal
./venv/bin/python train.py
```

### Inspect the Domain List

```bash
# Query ML-added exact blocks
docker exec pihole sqlite3 /etc/pihole/gravity.db \
  "SELECT domain, comment FROM domainlist WHERE comment LIKE '%ML Detector%';"

# Query ML-added wildcard regexes
docker exec pihole sqlite3 /etc/pihole/gravity.db \
  "SELECT domain, comment FROM domainlist WHERE type = 3 AND comment LIKE '%Wildcard Aggregation%';"
```

### Check Model File

```bash
# Verify the ONNX model exists and is valid
ls -lh ./model.onnx
```

---

## Edge Cases & Troubleshooting

### Permissions Errors

**Docker socket access denied:**

The detector requires the Docker socket to invoke `pihole reloaddns`. Ensure the container has access:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

If the Pi-hole container is named something other than `pihole`, the `docker exec` calls in `detector.py` and `sweep.py` will fail. Modify the container name in the scripts or create a symlink.

**SQLite database locked:**

Both `detector.py` and `sweep.py` use read-only URI connections (`?mode=ro`) for initial reads to avoid locking conflicts with Pi-hole's FTL daemon. Write operations use standard connections with a 20-second timeout. If you encounter lock errors, check that no other process is holding a write lock on `gravity.db`.

### False Positives

If a legitimate domain is incorrectly blocked:

1. **Check the blocklist:**
   ```bash
   docker exec pihole sqlite3 /etc/pihole/gravity.db \
     "SELECT * FROM domainlist WHERE domain = 'example.com';"
   ```

2. **Remove the block:**
   ```bash
   docker exec pihole sqlite3 /etc/pihole/gravity.db \
     "DELETE FROM domainlist WHERE domain = 'example.com';"
   docker exec pihole pihole reloaddns
   ```

3. **Add a permanent allowlist entry** to prevent re-classification:
   ```bash
   docker exec pihole sqlite3 /etc/pihole/gravity.db \
     "INSERT INTO domainlist (type, domain, enabled, date_added, date_modified, comment)
      VALUES (0, 'example.com', 1, $(date +%s), $(date +%s), 'Whitelisted by admin');"
   docker exec pihole pihole reloaddns
   ```

### False Negatives (Ads Not Blocked)

If ad domains are slipping through:

1. **Check confidence threshold** — The 0.95 threshold is intentionally high. Lower it in `detector.py` if you want more aggressive blocking (at the risk of more false positives):
   ```python
   CONFIDENCE_THRESHOLD = 0.90  # More aggressive
   ```

2. **Check bypass lists** — The domain may be in the Majestic Top 100k or matching a CDN suffix. Review the bypass logs in the detector output.

3. **Retrain the model** — If the ad landscape has shifted, retrain on fresh data:
   ```bash
   ./venv/bin/python train.py
   ```

### DGA Wildcard Too Broad

If a wildcard regex is blocking too many subdomains:

1. **Identify the regex:**
   ```bash
   docker exec pihole sqlite3 /etc/pihole/gravity.db \
     "SELECT domain FROM domainlist WHERE type = 3 AND comment LIKE '%Wildcard%';"
   ```

2. **Remove the regex:**
   ```bash
   docker exec pihole sqlite3 /etc/pihole/gravity.db \
     "DELETE FROM domainlist WHERE type = 3 AND domain LIKE '%your-root-domain%';"
   docker exec pihole pihole reloaddns
   ```

3. **Add specific subdomains as allowlist entries** if needed.

### Model File Not Found

If `detector.py` reports `Model file not found`:

- **Containerized:** Ensure `model.onnx` is mapped into the container at `/app/model.onnx`
- **Bare-metal:** Ensure `model.onnx` exists in the script's directory. Run `train.py` first.

### FTL Database Empty or Missing

If `detector.py` reports `No recent domains found`:

- Verify Pi-hole is actively resolving queries
- Check that `pihole-FTL.db` is accessible at the configured path
- Ensure the database contains query records with status 2 (forwarded) or 3 (cached)

### Cron Jobs Not Executing (Bare-Metal)

```bash
# Verify cron is running
systemctl status cron

# Check cron permissions (files must be 644, directories 755)
chmod 644 /etc/cron.d/ml-detector 2>/dev/null

# Check cron logs
grep CRON /var/log/syslog | tail -20
```

---

## Project Structure

```
pihole-llm/
├── detector.py          # Real-time domain classifier (runs every 5 min)
├── sweep.py             # ML blocklist consolidation (runs daily at 3 AM)
├── train.py             # Model retraining (runs monthly on the 1st)
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # Container image definition
├── requirements.txt     # Python dependencies
├── setup.sh             # Bare-metal environment setup script
├── model.onnx           # Trained ONNX model (generated by train.py)
├── LICENSE              # MIT License
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
