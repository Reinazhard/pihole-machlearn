# Pi-hole ML Ad/Tracker Detector

A zero-latency, on-device Machine Learning DNS sinkhole that connects directly to Pi-hole databases.

## 1. Train the ML Model
The Char-CNN neural network must be trained on your current `gravity.db` (ads) and the Majestic Million list (safe domains) to learn what trackers look like structurally.
This takes about 2-5 minutes to complete on a VPS CPU.

```bash
cd /home/haru/pihole-llm
./venv/bin/python train.py
```
*This will output a `model.onnx` file.*

## 2. Enable the Background Scanner
The ML Detector is deployed as a Docker container alongside your Pi-hole stack.

To start it:
```bash
cd /opt/haru/dns-stack
docker-compose up -d --build ml-detector
```

The container uses an internal `cron` daemon to automatically run:
1. **Detector:** Every 5 minutes (Instantly evaluates new domains).
2. **Sweeper:** Daily at 3:00 AM (Moves individual Pi-hole blocks into `ml-blocklist.txt` to keep the UI clean).
3. **Trainer:** Monthly on the 1st at 2:00 AM (Downloads fresh Majestic Million safe domains, pulls new Ad domains from Pi-hole, and retrains the AI model to recognize new tracking patterns).

## 3. Maintenance
- **Checking Logs:** Monitor what the ML model is doing by viewing the docker logs:
  `docker logs -f ml-detector`
- **Whitelisting:** If the model accidentally blocks a service you use, edit the `whitelist_keywords` array in `/home/haru/pihole-llm/detector.py` and rebuild the container.
- **Manual Retraining:** If you want to force a retraining outside of the monthly schedule, run:
  `docker exec ml-detector python train.py`
