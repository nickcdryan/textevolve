import os
import re
import math

# Hypothesis: Answering factual questions using a LLM-driven QA system with dual knowledge base lookups (KB1 and KB2) and hierarchical answer validation where KB1 has greater influence and can override KB2.
# We will use two simulated knowledge bases, KB1 acting as a "primary" KB with greater authority, and KB2 as a "secondary" KB. We'll implement an LLM validator that trusts KB1 more than KB2 when there's a conflict.
# This is designed to test how hierarchical knowledge source influence impacts overall accuracy and robustness.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"





def main(question):
    answer = call_llm(question)
    return answer
