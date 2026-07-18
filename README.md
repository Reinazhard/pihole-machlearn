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
Once the model is trained, you need to enable the cron jobs. 
1. The **Detector** runs every 5 minutes to instantly block new trackers.
2. The **Sweeper** runs once a day at 3:00 AM to move those individual blocks into a single Pi-hole Adlist file, keeping your Pi-hole web UI perfectly clean.

Run this command in your terminal to enable both:
```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * cd /home/haru/pihole-llm && ./venv/bin/python detector.py >> detector.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "0 3 * * * cd /home/haru/pihole-llm && ./venv/bin/python sweep.py >> sweep.log 2>&1") | crontab -
```

## 3. Maintenance
- **Checking Logs:** You can monitor what the ML model is doing by viewing the log file:
  `tail -f /home/haru/pihole-llm/detector.log`
- **Whitelisting:** If the model accidentally blocks a service you use, edit the `whitelist_keywords` array in `/home/haru/pihole-llm/detector.py` and remove the domain from your Pi-hole web UI.
- **Retraining:** Run `train.py` every few months to teach the model new tracker patterns.
