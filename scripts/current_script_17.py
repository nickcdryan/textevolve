import os
import re
import math

# New Approach: Structured Decomposition and Fact Verification with Adaptive Querying
# Hypothesis: Decomposing the question into structured components, using those components to generate adaptive search queries, and then verifying extracted facts against the original question structure will improve accuracy.
# This approach combines structured analysis with flexible querying.

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

def main(question, max_attempts=3):
    """Solve factual questions using Structured Decomposition and Fact Verification with Adaptive Querying."""

    # Step 1: Structured Question Decomposition
    decomposition_prompt = f"""
    Decompose the question into these structured components:
    - Entities: The key objects or concepts in the question.
    - Attributes: The specific properties or characteristics being asked about.
    - Constraints: Any limitations or conditions that must be met.

    Example:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Decomposition:
    {{
        "Entities": ["Great Barrier Reef"],
        "Attributes": ["capital"],
        "Constraints": ["country"]
    }}

    Question: {question}
    Decomposition:
    """
    decomposition_result = call_llm(decomposition_prompt, system_instruction="You are an expert at structured question decomposition.")

    # Step 2: Adaptive Query Generation
    query_prompt = f"""
    Generate a search query based on the decomposed question. Adapt the query to include the entities, attributes, and constraints.

    Example:
    Decomposition:
    {{
        "Entities": ["Great Barrier Reef"],
        "Attributes": ["capital"],
        "Constraints": ["country"]
    }}
    Query: "capital of country with Great Barrier Reef"

    Decomposition: {decomposition_result}
    Query:
    """
    search_query = call_llm(query_prompt, system_instruction="You are an expert at adaptive query generation.")

    # Step 3: Information Retrieval (simulated)
    search_results = call_llm(f"Simulated web search results for: {search_query}. Focus on concise and relevant results.", "You are a helpful search engine.")

    # Step 4: Fact Extraction
    extraction_prompt = f"""
    Extract a concise answer from the search results.

    Example:
    Search Results: Canberra is the capital of Australia.
    Answer: Canberra

    Search Results: {search_results}
    Answer:
    """
    extracted_answer = call_llm(extraction_prompt, system_instruction="You are an expert at concise answer extraction.")

    # Step 5: Fact Verification
    verification_prompt = f"""
    Verify that the extracted answer is a correct and complete answer to the original question, taking into account the original question's entities, attributes, and constraints from the decomposed question.

    Example:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Answer: Canberra
    Verification: VALID - Canberra is the capital of Australia, where the Great Barrier Reef is located.

    Question: {question}
    Answer: {extracted_answer}
    Verification:
    """

    verification_result = call_llm(verification_prompt, system_instruction="You are a strict fact verifier.")
    print(f"Verification Result: {verification_result}") #Print to see if the last step validated and why
    
    if "VALID" in verification_result:
        return extracted_answer
    else:
        return "Could not be validated."