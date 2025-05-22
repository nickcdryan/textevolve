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

def simulate_knowledge_base(query, kb_id):
    """Simulate retrieving information from a knowledge base."""
    system_instruction = f"You are a simulated knowledge base with ID {kb_id}. Provide factual information relevant to the query. KB1 is extremely factual and detail driven, rigorously checking its knowledge. KB2 relies heavily on reasoning. KB3 retrieves all surrounding facts, even distantly related facts, before making a decision. KB4 takes a holistic approach, combining the previous approaches."
    prompt = f"""
    Simulate retrieving information based on this query.

    Example 1 (KB):
    Query: capital of France
    Result: Paris is the capital of France.

    Query: {query}
    Result:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, kb1_result, kb2_result, kb3_result, kb4_result):
    """Extract the answer from the knowledge base results, with KB1 taking precedence."""
    system_instruction = "You are an expert at extracting answers from knowledge base results."
    prompt = f"""
    Extract the best answer to the question, considering the information from four knowledge bases. Choose the answer with the greatest majority.

    Question: {question}
    KB1 Result: {kb1_result}
    KB2 Result: {kb2_result}
    KB3 Result: {kb3_result}
    KB4 Result: {kb4_result}
    Answer:
    """
    return call_llm(prompt, system_instruction)



def main(question):
    """Solve questions using dual knowledge base lookups and validation."""
    try:
        # Simulate retrieval from KB1
        kb1_result = simulate_knowledge_base(question, "KB1")

        # Simulate retrieval from KB2
        kb2_result = simulate_knowledge_base(question, "KB2")
        kb3_result = simulate_knowledge_base(question, "KB3")
        kb4_result = simulate_knowledge_base(question, "KB4")

        # Extract the answer
        answer = extract_answer(question, kb1_result, kb2_result, kb3_result, kb4_result)

        return answer

    except Exception as e:
        return f"Error: {str(e)}"