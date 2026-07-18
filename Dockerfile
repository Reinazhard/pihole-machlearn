FROM python:3.10-slim

# Install cron and docker CLI (to allow docker exec on the socket)
RUN apt-get update && apt-get install -y cron curl && \
    curl -fsSL https://get.docker.com | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY detector.py sweep.py train.py ./

# Setup cron jobs
# 1. detector.py every 5 mins
# 2. sweep.py daily at 3:00 AM
# 3. train.py monthly at 2:00 AM on the 1st
RUN echo "*/5 * * * * cd /app && /usr/local/bin/python detector.py > /proc/1/fd/1 2>&1" > /etc/cron.d/ml-detector && \
    echo "0 3 * * * cd /app && /usr/local/bin/python sweep.py > /proc/1/fd/1 2>&1" >> /etc/cron.d/ml-detector && \
    echo "0 2 1 * * cd /app && /usr/local/bin/python train.py > /proc/1/fd/1 2>&1" >> /etc/cron.d/ml-detector && \
    chmod 0644 /etc/cron.d/ml-detector && \
    crontab /etc/cron.d/ml-detector

# Start cron in foreground
CMD ["cron", "-f"]
