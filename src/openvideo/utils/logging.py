import logging
import logging.config
import json
from pathlib import Path
from typing import Optional
import os

def setup_logging(
    default_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = "OPENVIDEO_LOG_CFG"
) -> None:
    """Setup logging configuration."""
    path = default_path or Path(__file__).parent / "logging.json"
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)
    
    if path.exists():
        with open(path) as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name) 