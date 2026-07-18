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

# Provide a dummy script to run initial training on container startup instead of build time
# It also dumps the Docker environment variables into /etc/environment so Cron can read them
RUN echo '#!/bin/bash\n\
env > /etc/environment\n\
if [ ! -f /app/data/model.onnx ]; then\n\
    echo "Running initial model training (this will take a few minutes)..."\n\
    /usr/local/bin/python /app/train.py\n\
fi\n\
echo "Starting cron daemon..."\n\
cron -f' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Setup cron jobs
# 1. detector.py every 5 mins
# 2. sweep.py daily at 3:00 AM
# 3. train.py monthly at 2:00 AM on the 1st
# We source /etc/environment to ensure cron inherits the Docker env vars (GRAVITY_DB, etc.)
RUN echo "*/5 * * * * root . /etc/environment && cd /app && /usr/local/bin/python detector.py > /proc/1/fd/1 2>&1" > /etc/cron.d/ml-detector && \
    echo "0 3 * * * root . /etc/environment && cd /app && /usr/local/bin/python sweep.py > /proc/1/fd/1 2>&1" >> /etc/cron.d/ml-detector && \
    echo "0 2 1 * * root . /etc/environment && cd /app && /usr/local/bin/python train.py > /proc/1/fd/1 2>&1" >> /etc/cron.d/ml-detector && \
    chmod 0644 /etc/cron.d/ml-detector

# Start via entrypoint script
CMD ["/app/entrypoint.sh"]
