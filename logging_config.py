import logging
import os

# Project packages; submodules (e.g. "agent.tools") inherit the level.
# "__main__" when running `python -m agent.agent`
APP_LOGGERS = ("__main__", "api", "agent", "domain", "retrieval", "storage", "ingest")


def setupLogging():
    """
    Configures logging once, at the entrypoint (app.py / agent.py __main__).

    LOG_LEVEL (.env, default INFO) applies only to the project's modules.
    Libraries stay at WARNING: botocore at DEBUG logs the AWS request
    signatures and httpx logs every OpenAI call.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(level)
