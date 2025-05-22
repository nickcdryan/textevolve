import os
import re
import math

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
    """Solve factual questions using a new approach: Active Information Retrieval and Contextual Answer Extraction with Multi-Example Prompting."""

    # Hypothesis: By actively seeking *multiple* sources of information with queries focused on the context of the question, along with a detailed, multi-example extraction prompt, we can improve answer accuracy. This strategy emphasizes contextual retrieval over broad concept expansion and uses multiple examples to show the LLM how to extract contextually relevant answers.

    # Step 1: Generate search queries focused on the question's context (with examples)
    contextual_query_prompt = f"""
    Generate multiple search queries to answer the question. Focus each query on a different aspect of the question, emphasizing the specific context. Provide 3 different queries.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Queries:
    1. "Mehbooba Mufti younger daughter name"
    2. "Mehbooba Mufti family"
    3. "Iltija Mufti biography"

    Example 2:
    Question: Which DropoutTV series is a spin-off of "Game Changer" inspired by its "Noise Boys" episodes?
    Queries:
    1. "DropoutTV series spin-off Game Changer Noise Boys"
    2. "Make Some Noise DropoutTV Game Changer"
    3. "DropoutTV Noise Boys spin-off series"

    Question: {question}
    Queries:
    """
    search_queries = call_llm(contextual_query_prompt, system_instruction="You are an expert query generator who formulates effective search queries.").split("\n")
    print(f"Search Queries: {search_queries}")

    # Step 2: Simulate Information Retrieval
    search_results = []
    for query in search_queries:
      search_results.append(call_llm(f"Simulated search results for: {query}. Focus on concise and relevant results.", "You are a search engine.")) #simulate a search engine
    print (f"Search results: {search_results}")

    # Step 3: Extract Answer from Contextualized Search Results (with multi-example prompting)
    answer_extraction_prompt = f"""
    Extract the concise answer to the original question from the following search results. Provide the *most* factually correct answer and the related search result.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Search Results:
    1. Iltija Mufti is the daughter of Mehbooba Mufti.
    2. Mehbooba Mufti has two daughters.
    Answer: Iltija Mufti.

    Example 2:
    Question: Which DropoutTV series is a spin-off of "Game Changer" inspired by its "Noise Boys" episodes?
    Search Results:
    1. Make Some Noise is a spin-off of Game Changer.
    2. Noise Boys inspired Make Some Noise.
    Answer: Make Some Noise.

    Example 3:
    Question: What year was the municipality of Santo Domingo, Antioquia, Colombia, founded?
    Search Results:
    1. Santo Domingo was founded in 1778.
    2. Santo Domingo is a municipality in Colombia.
    Answer: 1778.

    Question: {question}
    Search Results:
    {search_results}
    Answer:
    """
    extracted_answer = call_llm(answer_extraction_prompt, system_instruction="You are an answer extraction expert focusing on factually correct answers based on the search results.")
    print(f"Extracted Answer: {extracted_answer}")

    # Step 4: Validation (with example)
    validation_prompt = f"""
    Validate that the extracted answer correctly and completely answers the original question.

    Example 1:
    Question: What is the full name of the younger daughter of Mehbooba Mufti, a politician from Kashmir?
    Answer: Iltija Mufti.
    Validation: VALID - Iltija Mufti is the younger daughter of Mehbooba Mufti.

    Question: {question}
    Answer: {extracted_answer}
    Validation:
    """
    validation_result = call_llm(validation_prompt, system_instruction="You are a strict validator who checks extracted answers.").strip()

    if "VALID" in validation_result:
        return extracted_answer
    else:
        return "Could not be validated."