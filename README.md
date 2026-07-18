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

The container uses an internal `cron` daemon to automatically run the detector every 5 minutes, and the sweeper daily at 3:00 AM. 

## 3. Maintenance
- **Checking Logs:** Monitor what the ML model is doing by viewing the docker logs:
  `docker logs -f ml-detector`
- **Whitelisting:** If the model accidentally blocks a service you use, edit the `whitelist_keywords` array in `/home/haru/pihole-llm/detector.py` and rebuild the container.
- **Retraining:** Run `train.py` on the host to update `model.onnx`, then restart the container:
  `docker restart ml-detector`
