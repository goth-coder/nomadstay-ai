import json
import logging
import os
from datetime import UTC, datetime

from observability.context import currentRequestId

APP_LOGGERS = ("__main__", "api", "agent", "domain", "retrieval", "storage", "ingest", "observability")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        event = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),   
            "level": record.levelname,
            "logger": record.name,
            "request_id": currentRequestId(),
            "message": record.getMessage(),
        }
        event.update(getattr(record, "fields", {}))
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(event, ensure_ascii=False, default=str)                     


def setupLogging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(level)
