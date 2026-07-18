# Pi-hole ML Ad/Tracker Detector

Zero-latency, on-device ML DNS sinkhole. Classifies domains in real-time using a Char-CNN neural network, blocking ads and trackers before they resolve. No cloud, no API calls, no latency.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    ml-detector (Docker)                          │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐               │
│  │  detector   │   │   sweep    │   │   train    │               │
│  │  (5 min)    │   │  (daily)   │   │ (monthly)  │               │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘               │
│        │                 │                 │                      │
│        ▼                 ▼                 ▼                      │
│  ┌─────────────────────────────────────────────────┐             │
│  │              SQLite (gravity.db)                 │             │
│  │         Read-Only URI mode for safety            │             │
│  └─────────────────────────────────────────────────┘             │
│        │                                                         │
│        ▼                                                         │
│  ┌─────────────────────────────────────────────────┐             │
│  │           ONNX Runtime (CPU)                    │             │
│  │         Char-CNN binary classifier              │             │
│  │      Safe (0) ←──→ Ad/Tracker (1)              │             │
│  └─────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

## Deep Learning Model

**Architecture:** Character-level Convolutional Network (Char-CNN)

The model processes raw domain strings byte-by-byte, learning structural patterns that distinguish ad/tracker domains from legitimate ones. No hand-crafted features, no DNS lookups, no external data.

### Pipeline

1. **Encoding:** Each domain is truncated/padded to 100 bytes, stored as raw UTF-8 byte values
2. **Embedding:** 256-dim byte vocabulary → 32-dim dense vectors
3. **Convolution:** 3× Conv1D (128 filters, kernel=3) with MaxPool
4. **Classification:** FC(128×12 → 128) → Dropout(0.5) → FC(128 → 2)
5. **Output:** Softmax probabilities for `[Safe, Ad/Tracker]`

### Training

- **Dataset:** 150k ad domains from Pi-hole `gravity.db` + 150k safe domains from Majestic Million
- **Loss:** Weighted CrossEntropy (`class 0: 3.0, class 1: 1.0`) — penalizes false positives 3× harder
- **Export:** ONNX format for CPU inference via `onnxruntime`
- **Retraining:** Monthly, automatically via cron

### Alternatives Considered

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| TF-IDF + Logistic Regression | Fast, interpretable | Brittle to new TLDs, misses byte patterns | Deprecated |
| Char-CNN (current) | Captures subdomain structure, no feature engineering | Requires ONNX runtime | Active |
| Transformer-based | Better context | 100ms+ latency per domain | Overkill |

---

## Fail-safes & False Positive Mitigation

False positives in a sinkhole break real services. The system implements three defense layers:

### 1. Dynamic Top 100k Majestic Bypass

A read-only snapshot of the Majestic Million top domains acts as a strict allowlist. Any domain appearing in the top 100k by traffic rank is **never** classified, regardless of model output. Updated on each retrain cycle.

### 2. Selective CDN/Infrastructure Suffix Bypass

Known CDN and cloud infrastructure domains are bypassed via suffix matching:

```
.googleapis.com    .cloudfront.net    .amazonaws.com
.akamaihd.net      .fbcdn.net         .gstatic.com
.shopeemobile.com  .googleusercontent.com  .susercontent.com
.whatsapp.net
```

These are trust boundaries — blocking them breaks legitimate services. The suffix list is intentionally conservative and manually curated.

### 3. >95% Confidence Threshold

Only domains with `P(ad) > 0.95` are blocked. This is deliberately high. The model is trained to err on the side of letting ads through rather than blocking safe domains.

---

## DGA Wildcard Aggregation

When the ML detector identifies multiple suspicious subdomains under the same root domain (≥3 in the blocklist), it automatically upgrades from exact blocking to regex wildcard blocking.

**Logic:**
```python
# If 3+ subdomains of example.com are blocked:
#   → Insert regex: (\.|^)example\.com$
# This catches all current and future subdomains
```

**Example:**
```
ad1.tracking.com → blocked (exact)
ad2.tracking.com → blocked (exact)
ad3.tracking.com → aggregated → (\.|^)tracking\.com$ (wildcard)
ad4.tracking.com → caught by wildcard automatically
```

The wildcard is inserted into Pi-hole's `domainlist` (type 3) and takes effect on the next `gravity` update. Individual exact entries for aggregated roots are cleaned up by the sweeper.

---

## Components

### detector.py (Every 5 minutes)

The real-time classifier. Runs as the core pipeline:

1. Queries `pihole-FTL.db` (read-only) for domains allowed in the last 5 minutes
2. Filters out domains already in `gravity.db` blocklists
3. Applies Majestic Top 100k and CDN suffix bypass filters
4. Encodes remaining domains as byte sequences
5. Runs ONNX inference batch
6. Blocks domains exceeding 0.95 confidence
7. Triggers DGA wildcard aggregation
8. Calls `pihole reloaddns` via Docker socket

### sweep.py (Daily at 3:00 AM)

Cleans up the Pi-hole admin UI by migrating ML-added exact blocks to a consolidated adlist file:

1. Reads all ML-added exact blocks from `domainlist` (type 1, comment matching)
2. Appends them to `ml-blocklist.txt`
3. Removes the individual entries from `domainlist`
4. Registers `ml-blocklist.txt` in Pi-hole's `adlist` table
5. Runs `pihole -g` to compile the gravity database

### train.py (Monthly on the 1st at 2:00 AM)

Retrains the model on fresh data:

1. Downloads the latest Majestic Million safe domains
2. Pulls ad domains from the current `gravity.db`
3. Trains the Char-CNN (5 epochs, Adam optimizer)
4. Exports to `model.onnx`

---

## Deployment

### Prerequisites

- Docker and Docker Compose
- Pi-hole running in a Docker container (named `pihole`)
- Gravity database at `/etc/pihole/gravity.db`

### Step 1: Train the Model

```bash
cd pihole-llm
./venv/bin/python train.py
```

Produces `model.onnx`. Takes 2-5 minutes on CPU.

### Step 2: Build and Deploy

```bash
cd pihole-llm
docker-compose up -d --build ml-detector
```

### Step 3: Verify

```bash
docker logs -f ml-detector
```

You should see detector running every 5 minutes, processing domains.

### Standalone Deployment (External)

The `docker-compose.yml` in this repo is self-contained. To deploy alongside an existing Pi-hole stack elsewhere:

```bash
# Copy the files
scp -r ./pihole-llm user@host:/opt/ml-detector

# Adjust paths in docker-compose.yml volumes, then:
docker-compose up -d --build ml-detector
```

The container expects:
- Pi-hole container named `pihole` on the `dns` network
- `gravity.db` and `pihole-FTL.db` mounted at the expected paths
- Docker socket for `reloaddns` calls

---

## Database Resilience

All read operations use **SQLite URI mode with read-only flag**:

```python
ro_uri = f"file:{GRAVITY_DB}?mode=ro"
conn = sqlite3.connect(ro_uri, uri=True, timeout=20.0)
```

This prevents the detector from corrupting Pi-hole's databases during concurrent writes (e.g., when Pi-hole processes DNS queries or runs its own updates). Write operations are isolated to their own connections with explicit commit/rollback.

---

## Manual Operations

| Command | Description |
|---------|-------------|
| `docker exec ml-detector python detector.py` | Run detection manually |
| `docker exec ml-detector python sweep.py` | Force sweep now |
| `docker exec ml-detector python train.py` | Retrain model manually |
| `docker logs -f ml-detector` | Monitor real-time logs |
| `docker restart ml-detector` | Restart the service |

---

## File Structure

```
pihole-llm/
├── detector.py          # Real-time ML domain classifier
├── sweep.py             # Daily blocklist aggregator
├── train.py             # Monthly model trainer
├── Dockerfile           # Container with cron scheduler
├── docker-compose.yml   # Standalone deployment
├── requirements.txt     # Python dependencies
├── setup.sh             # Local venv setup (dev only)
├── model.onnx           # Trained ONNX model (generated)
└── README.md
```
