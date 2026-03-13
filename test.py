import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests

# Simple local agentic AI app using Ollama + Llama 3
# 1) Install Ollama
# 2) Pull a model, for example:
#       ollama pull llama3
# 3) Run this file:
#       python simple_agentic_ai_app.py
#
# This app uses a local LLM to:
# - decide whether to answer directly or use a tool
# - call one tool
# - turn the tool result into a final answer

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:latest"


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[dict], Any]


def get_weather(args: dict) -> str:
    city = args.get("city", "unknown city")
    return f"Weather in {city}: warm, 30C, slight wind."


def add_numbers(args: dict) -> str:
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    return f"The sum of {a} and {b} is {a + b}."


def save_note(args: dict) -> str:
    note = args.get("note", "")
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(note + "\n")
    return f"Saved note: {note}"


def read_notes(args: dict) -> str:
    if not os.path.exists("notes.txt"):
        return "No notes found yet."
    with open("notes.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content or "No notes found yet."


TOOLS: Dict[str, Tool] = {
    "get_weather": Tool(
        name="get_weather",
        description="Get the weather for a city. Input: {city}",
        func=get_weather,
    ),
    "add_numbers": Tool(
        name="add_numbers",
        description="Add two numbers. Input: {a, b}",
        func=add_numbers,
    ),
    "save_note": Tool(
        name="save_note",
        description="Save a note for later. Input: {note}",
        func=save_note,
    ),
    "read_notes": Tool(
        name="read_notes",
        description="Read all saved notes. Input: {}",
        func=read_notes,
    ),
}


PLANNER_PROMPT = """
You are a simple local AI agent.
Your job is to decide whether to answer directly or call exactly one tool.

Available tools:
- get_weather: Get the weather for a city. Input: {city}
- add_numbers: Add two numbers. Input: {a, b}
- save_note: Save a note for later. Input: {note}
- read_notes: Read all saved notes. Input: {}

Return ONLY valid JSON in one of these formats:

1) To use a tool:
{"action":"tool","tool_name":"get_weather","arguments":{"city":"Manila"}}

2) To answer directly:
{"action":"final","answer":"your answer here"}
""".strip()


FINALIZER_PROMPT = """
You are a helpful assistant.
The user asked something, a tool was run, and now you must answer clearly.
Be concise and useful.
""".strip()


class Agent:
    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    def ollama_chat(self, messages):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )

        print("STATUS:", response.status_code)
        print("BODY:", response.text)

        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    def llm_plan(self, user_input: str) -> dict:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_input},
        ]

        raw = self.ollama_chat(messages).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            lowered = user_input.lower()
            if "weather" in lowered:
                city = user_input.split("in")[-1].strip().rstrip("?") if "in" in lowered else "Manila"
                return {"action": "tool", "tool_name": "get_weather", "arguments": {"city": city}}
            if "add" in lowered or "+" in lowered:
                numbers = [float(x) for x in user_input.replace("+", " ").split() if x.replace(".", "", 1).isdigit()]
                if len(numbers) >= 2:
                    return {
                        "action": "tool",
                        "tool_name": "add_numbers",
                        "arguments": {"a": numbers[0], "b": numbers[1]},
                    }
            if "save note" in lowered or "remember this" in lowered:
                note = user_input.split(":", 1)[-1].strip()
                return {"action": "tool", "tool_name": "save_note", "arguments": {"note": note}}
            if "read notes" in lowered or "show notes" in lowered:
                return {"action": "tool", "tool_name": "read_notes", "arguments": {}}
            return {"action": "final", "answer": raw}

    def finalize(self, user_input: str, tool_result: str) -> str:
        messages = [
            {"role": "system", "content": FINALIZER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_input}\n"
                    f"Tool result: {tool_result}\n\n"
                    "Write the final answer for the user."
                ),
            },
        ]
        return self.ollama_chat(messages).strip()

    def run(self, user_input: str) -> str:
        plan = self.llm_plan(user_input)

        if plan.get("action") == "final":
            return plan.get("answer", "I could not answer that.")

        if plan.get("action") == "tool":
            tool_name = plan.get("tool_name")
            arguments = plan.get("arguments", {})

            tool = TOOLS.get(tool_name)
            if not tool:
                return f"Unknown tool: {tool_name}"

            tool_result = str(tool.func(arguments))
            return self.finalize(user_input, tool_result)

        return "Planner returned an invalid action."


if __name__ == "__main__":
    agent = Agent()
    print("Simple Llama 3 Agentic AI App")
    print("Make sure Ollama is running and llama3 is pulled.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            result = agent.run(user_input)
            print(f"Agent: {result}\n")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Start Ollama first.\n")
        except Exception as e:
            print(f"Error: {e}\n")
