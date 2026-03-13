"""
ARCHITECTURE: SIMPLE AGENTIC RAG WITH OLLAMA + LLAMA 3
======================================================

HIGH-LEVEL FLOW
---------------
User
  ↓
Planner (LLM)
  ↓
Decide next action
  ├─ retrieve_docs
  ├─ save_note
  ├─ read_notes
  └─ final answer

If retrieve_docs was chosen:
  ↓
Retriever tool
  ↓
Return top matching chunks
  ↓
Reviewer (LLM)
  ↓
Enough context?
  ├─ yes → Finalizer (LLM) → answer
  └─ no  → refine query → retrieve again

MAIN COMPONENTS
---------------
1) DOCUMENTS
   - Tiny local knowledge base.
   - In a real system this would come from PostgreSQL + pgvector.

2) TOOLS
   - retrieve_docs: does retrieval
   - save_note: stores a note
   - read_notes: reads notes

3) RETRIEVER
   - Current version uses simple bag-of-words cosine similarity.
   - Real version should use embeddings + pgvector similarity search.

4) PLANNER
   - Llama 3 decides whether to retrieve, use another tool, or answer directly.

5) REVIEWER
   - Llama 3 checks whether retrieved context is enough.
   - If not enough, it asks for another retrieval using a refined query.

6) FINALIZER
   - Llama 3 produces the final grounded answer using retrieved/tool output.

REAL-WORLD PGVECTOR VERSION
---------------------------
User query
  ↓
Embed query
  ↓
Search pgvector
  ↓
Get chunks + metadata
  ↓
Reviewer decides enough or retrieve again
  ↓
Finalizer answers

WHY THIS IS AGENTIC
-------------------
Normal RAG:
  query → retrieve once → answer

Agentic RAG:
  query → plan → retrieve → review → maybe retrieve again → answer

So the agentic part is not just retrieval.
It is the LLM deciding the next step.
"""

import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Any

import requests

# Simple Agentic RAG using Ollama (llama3:latest)
# -------------------------------------------------
# Install:
#   pip install requests
# Run Ollama first, then make sure the model exists:
#   ollama pull llama3
# Run:
#   python simple_agentic_rag_llama3.py
#
# What this demonstrates:
# 1) A tiny knowledge base
# 2) Simple retrieval via bag-of-words cosine similarity
# 3) An agent that decides whether to retrieve, use a tool, or answer directly
# 4) Llama 3 as the planner + final answer generator

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:latest"
TOP_K = 3


# -----------------------------
# Local knowledge base for RAG
# -----------------------------
DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "The secretary agent can schedule meetings, save notes, summarize text, and answer questions using retrieval.",
        "source": "capabilities"
    },
    {
        "id": "doc_2",
        "text": "RAG means Retrieval-Augmented Generation. First retrieve relevant context, then pass it to the language model to produce a grounded answer.",
        "source": "rag_basics"
    },
    {
        "id": "doc_3",
        "text": "Agentic RAG adds reasoning and decisions. The model can choose to retrieve again, use tools, validate answers, or ask follow-up questions.",
        "source": "agentic_rag"
    },
    {
        "id": "doc_4",
        "text": "A command-based assistant can map user intent to actions such as create_task, schedule_meeting, save_note, and inquiry.",
        "source": "commands"
    },
    {
        "id": "doc_5",
        "text": "Good RAG systems store metadata with chunks, such as source, title, tags, and action type, to support filtering and grounded responses.",
        "source": "metadata"
    },
]


# -----------------------------
# Tools
# -----------------------------
@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[dict], Any]


def save_note(args: dict) -> str:
    note = args.get("note", "")
    with open("agent_notes.txt", "a", encoding="utf-8") as f:
        f.write(note + "\n")
    return f"Saved note: {note}"


def read_notes(args: dict) -> str:
    try:
        with open("agent_notes.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content or "No notes found yet."
    except FileNotFoundError:
        return "No notes found yet."


def retrieve_docs(args: dict) -> str:
    query = args.get("query", "")
    top_k = int(args.get("top_k", TOP_K))
    results = search_documents(query, DOCUMENTS, top_k=top_k)

    if not results:
        return "No relevant documents found."

    lines = []
    for item in results:
        lines.append(
            f"[score={item['score']:.3f}] id={item['doc']['id']} source={item['doc']['source']} text={item['doc']['text']}"
        )
    return "".join(lines)


TOOLS: Dict[str, Tool] = {
    "retrieve_docs": Tool(
        name="retrieve_docs",
        description="Retrieve relevant knowledge base documents. Input: {query, top_k}",
        func=retrieve_docs,
    ),
    "save_note": Tool(
        name="save_note",
        description="Save a note for later. Input: {note}",
        func=save_note,
    ),
    "read_notes": Tool(
        name="read_notes",
        description="Read saved notes. Input: {}",
        func=read_notes,
    ),
}


# -----------------------------
# Tiny embedding-like retrieval
# -----------------------------

def tokenize(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def term_frequency(tokens: List[str]) -> Dict[str, float]:
    freq: Dict[str, float] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0.0) + 1.0
    return freq


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_documents(query: str, documents: List[dict], top_k: int = 3) -> List[dict]:
    query_vec = term_frequency(tokenize(query))
    scored = []

    for doc in documents:
        doc_vec = term_frequency(tokenize(doc["text"]))
        score = cosine_similarity(query_vec, doc_vec)
        scored.append({"doc": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [item for item in scored[:top_k] if item["score"] > 0]


# -----------------------------
# Llama prompts
# -----------------------------
PLANNER_PROMPT = """
You are an agentic RAG planner.
You must decide the next action for the user's request.

Available tools:
- retrieve_docs: retrieve relevant knowledge base documents. Input: {query, top_k}
- save_note: save a note. Input: {note}
- read_notes: read notes. Input: {}

Rules:
- If the user asks about knowledge, concepts, system design, policies, or information from the knowledge base, prefer retrieve_docs.
- If the user asks to save something, use save_note.
- If the user asks to show or read notes, use read_notes.
- Otherwise answer directly.
- Call exactly one tool at a time.
- Return ONLY valid JSON.

Formats:
1) Tool call:
{"action":"tool","tool_name":"retrieve_docs","arguments":{"query":"what is agentic rag","top_k":3}}

2) Final answer:
{"action":"final","answer":"your answer"}
""".strip()

REVIEW_PROMPT = """
You are a retrieval reviewer in a simple agentic RAG system.
You will inspect the user's request and the retrieved context.
Decide whether the retrieved context is enough.

Return ONLY valid JSON in one of these forms:
1) If enough:
{"decision":"answer"}

2) If not enough and another retrieval attempt is needed:
{"decision":"retrieve_again","refined_query":"better search query here"}
""".strip()

FINALIZER_PROMPT = """
You are a helpful assistant.
Use the tool result as grounded context.
If the tool result is retrieval output, answer based mainly on it.
Do not invent facts that are not supported by the tool result when the user asked a knowledge question.
""".strip()


class AgenticRAG:
    def ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        data = response.json()
        return data["message"]["content"]

    def plan(self, user_input: str) -> dict:
        raw = self.ollama_chat([
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_input},
        ]).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            lowered = user_input.lower()
            if any(word in lowered for word in ["rag", "agent", "retrieval", "secretary", "knowledge", "policy", "system"]):
                return {
                    "action": "tool",
                    "tool_name": "retrieve_docs",
                    "arguments": {"query": user_input, "top_k": TOP_K},
                }
            if "save" in lowered and "note" in lowered:
                return {
                    "action": "tool",
                    "tool_name": "save_note",
                    "arguments": {"note": user_input},
                }
            if "read" in lowered and "note" in lowered:
                return {
                    "action": "tool",
                    "tool_name": "read_notes",
                    "arguments": {},
                }
            return {"action": "final", "answer": raw}

    def review_retrieval(self, user_input: str, retrieved_context: str) -> dict:
        raw = self.ollama_chat([
            {"role": "system", "content": REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_input}"
                    f"Retrieved context: {retrieved_context}"
                    "Decide if this is enough to answer the user."
                ),
            },
        ]).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if "No relevant documents found" in retrieved_context:
                return {"decision": "retrieve_again", "refined_query": user_input}
            return {"decision": "answer"}

    def finalize(self, user_input: str, tool_result: str) -> str:
        return self.ollama_chat([
            {"role": "system", "content": FINALIZER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_input}"
                    f"Tool result: {tool_result}"
                    "Write the final answer for the user."
                ),
            },
        ]).strip()

    def run(self, user_input: str) -> str:
        plan = self.plan(user_input)
        print("[PLAN]", json.dumps(plan, indent=2))

        if plan.get("action") == "final":
            return plan.get("answer", "I could not answer that.")

        if plan.get("action") == "tool":
            tool_name = plan.get("tool_name")
            arguments = plan.get("arguments", {})
            tool = TOOLS.get(tool_name)

            if not tool:
                return f"Unknown tool: {tool_name}"

            if tool_name != "retrieve_docs":
                tool_result = str(tool.func(arguments))
                print("[TOOL RESULT]" + tool_result)
                return self.finalize(user_input, tool_result)

            current_query = str(arguments.get("query", user_input))
            max_loops = 2
            collected_results: List[str] = []

            for step in range(max_loops):
                tool_result = str(tool.func({"query": current_query, "top_k": TOP_K}))
                collected_results.append(f"Retrieval step {step + 1} using query: {current_query}{tool_result}")
                print("[TOOL RESULT]" + collected_results[-1])

                review = self.review_retrieval(user_input, tool_result)
                print("[REVIEW]", json.dumps(review, indent=2))

                if review.get("decision") == "answer":
                    break

                if review.get("decision") == "retrieve_again":
                    current_query = review.get("refined_query", current_query)
                    continue

                break

            merged_context = "".join(collected_results)
            return self.finalize(user_input, merged_context)

        return "Planner returned an invalid action."


if __name__ == "__main__":
    app = AgenticRAG()
    print("Simple Agentic RAG with Ollama + Llama 3")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            answer = app.run(user_input)
            print("\nAssistant:", answer, "\n")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
