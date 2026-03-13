import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:latest"

# for this architecture we are going to use just one model
class agentic_rag():
    def __init__(self, query, foundation_model):
        self.query = query
        self.foundation_model = foundation_model
        self.context = ""

    def model(self):
        ## model
        passed = False
        self.context = self.query

        while passed == False:
            decision = self.foundation_model(self.context)

            if decision == 'plan':
                self.context = ''
                print("vector storage")

            elif decision == 'review':
                self.context = ''
                print("vector storage")

            else:
                print("finalize")
                passed = True

        return passed

    def run(self):
        return self.model()


if __name__ == "__main__":
    user_input = input("You: ").strip()

    # placeholder model function (since OLLAMA_MODEL is just a string)
    def foundation_model(context):
        return "finalize"

    agent = agentic_rag(user_input, foundation_model)
    output = agent.run()

    print(f"agent: {output}")