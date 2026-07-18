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
Once the model is trained, you need to enable the cron job. This cron job runs every 5 minutes in the background. It reads the last 5 minutes of allowed queries from Pi-hole, evaluates them with the ONNX model, blocks any new trackers, and reloads Pi-hole.

Run this command in your terminal to enable it:
```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * cd /home/haru/pihole-llm && ./venv/bin/python detector.py >> detector.log 2>&1") | crontab -
```

## 3. Maintenance
- **Checking Logs:** You can monitor what the ML model is doing by viewing the log file:
  `tail -f /home/haru/pihole-llm/detector.log`
- **Whitelisting:** If the model accidentally blocks a service you use, edit the `whitelist_keywords` array in `/home/haru/pihole-llm/detector.py` and remove the domain from your Pi-hole web UI.
- **Retraining:** Run `train.py` every few months to teach the model new tracker patterns.
