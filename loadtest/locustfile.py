"""
Simulated nomads: ask, wait, sometimes ask a follow-up in the same session.

    uv run locust -f loadtest/locustfile.py --host http://localhost:8000
    # opens http://localhost:8089: choose users and spawn rate

Run the API with MOCK_LLM=1 unless you mean to pay for every request.
"""

import os
import random

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv(".env")

QUESTIONS = [
    "Porto under 100 euros, quiet, I have video calls all day.",
    "I need a proper desk to work. Porto, max 80 EUR a night.",
    "Quiet place in Porto near a gym, under 120 EUR.",
    "What's the best place in Porto for remote work?",
    "Somewhere well connected by metro in Porto, under 150 EUR.",
]
FOLLOW_UPS = ["Tell me more about the second one.", "Is the host responsive?"]


class Nomad(HttpUser):
    wait_time = between(1, 5)  # a person reads the answer before asking again

    def on_start(self):
        self.headers = {"X-API-Key": os.environ["API_KEY"]}
        self.session_id = None

    @task(3)
    def ask(self):
        response = self.client.post("/agent/run", headers=self.headers, name="/agent/run",
                                    json={"message": random.choice(QUESTIONS)})
        if response.ok:
            self.session_id = response.json()["session_id"]

    @task(1)
    def follow_up(self):
        if self.session_id is None:
            return
        self.client.post("/agent/run", headers=self.headers, name="/agent/run (follow-up)",
                         json={"message": random.choice(FOLLOW_UPS), "session_id": self.session_id})
