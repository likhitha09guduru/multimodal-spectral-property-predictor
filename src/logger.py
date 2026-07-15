import logging
import os
import sys
from datetime import datetime

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

handlers = [logging.StreamHandler(sys.stdout)]

# File logging is opt-in (off by default) so container deployments -- where
# stdout is what gets collected -- don't need a writable volume just to boot.
if os.environ.get("LOG_TO_FILE", "false").lower() == "true":
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    handlers.append(logging.FileHandler(os.path.join(logs_dir, LOG_FILE)))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
