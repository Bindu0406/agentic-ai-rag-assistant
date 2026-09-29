import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the query function from src/graph.py
from src.graph import query_agent

TEST_QUERIES = [
    "What are the core components of agentic AI according to the book?",
    "How does an agent differ from a standard LLM prompt pipeline?",
    "What role does planning play in autonomous decision-making?",
    "Explain how memory mechanisms function within an agent architecture.",
    "What are common design patterns used for multi-agent collaboration?",
    "What is the capital of France and what is its population?"  # Out-of-domain / Guardrail check
]

def run_tests():
    print("=" * 70)
    print("RUNNING BENCHMARK EVALUATION TEST SUITE (AGENTIC AI RAG)")
    print("=" * 70 + "\n")

    for idx, question in enumerate(TEST_QUERIES, 1):
        print(f"[{idx}/{len(TEST_QUERIES)}] Query: {question}")
        print("-" * 70)
        
        try:
            answer = query_agent(question)
            print(f"Generated Grounded Response:\n{answer}\n")
        except Exception as e:
            print(f"Error executing query: {e}\n")
            
        print("=" * 70 + "\n")

if __name__ == "__main__":
    run_tests()