#!/usr/bin/env python
"""
Test script to verify LLM is working properly
"""
import os
from system_tools import call_llm

# Simple test
prompt = """
Extract numbers from this question and return as JSON.

Question: "What is 10 plus 5?"

Return format:
{
  "numbers": [10, 5],
  "operation": "add"
}

Output:
"""

system_instruction = "You are a data extraction expert. Return only valid JSON, no explanations."

print("Testing call_llm...")
print(f"Prompt: {prompt[:100]}...")
print(f"System instruction: {system_instruction}")
print("\nCalling LLM...")

result = call_llm(prompt, system_instruction)

print(f"\nResult type: {type(result)}")
print(f"Result length: {len(result) if result else 0}")
print(f"Result content: {repr(result[:500])}")
print(f"\nFirst 200 chars: {result[:200]}")

