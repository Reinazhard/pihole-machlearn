#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$DIR/venv"

echo "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -r "$DIR/requirements.txt"

echo "Environment ready."
echo ""
echo "To train the model, run:"
echo "$VENV_DIR/bin/python $DIR/train.py"
echo ""
echo "To test the detector manually, run:"
echo "$VENV_DIR/bin/python $DIR/detector.py"
echo ""
echo "To set up the cron job to run every 5 minutes, run this command:"
echo "(crontab -l 2>/dev/null; echo \"*/5 * * * * $VENV_DIR/bin/python $DIR/detector.py >> $DIR/detector.log 2>&1\") | crontab -"
