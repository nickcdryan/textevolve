import os
import re
import math

# Hypothesis: Leveraging a layered validation approach with specific validation agents to address fact retrieval and date handling weaknesses.
# This script introduces a new layered validation system with specialized validator agents for fact-checking, temporal consistency, and source reliability.

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

def simulate_search(query, engine_id):
    """Simulate different search engines."""
    system_instruction = f"You are a simulated search engine with ID {engine_id} providing factual and CONCISE information. You MUST provide a verifiable source URL at the end of your answer. Be concise."
    prompt = f"""
    Simulate search results for the query: '{query}'.

    Example 1 (Engine ID: 1, Source: Wikipedia):
    Query: capital of Australia
    Search Results: Canberra is the capital of Australia. Source: en.wikipedia.org/wiki/Canberra

    Example 2 (Engine ID: 2, Source: Britannica):
    Query: capital of Australia
    Search Results: Australia's capital is Canberra, located in the Australian Capital Territory. Source: britannica.com/place/Canberra

    Query: {query}
    Search Results:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, search_results):
    """Extract potential answers from search results."""
    system_instruction = "You are an answer extraction expert, focusing on precision. You MUST extract the concise answer and the source URL from the provided search results."
    prompt = f"""
    Extract the concise answer and its source from the search results.

    Example:
    Question: What is the capital of Australia?
    Search Results: Canberra is the capital of Australia. Source: en.wikipedia.org/wiki/Canberra
    Answer: Canberra Source: en.wikipedia.org/wiki/Canberra

    Question: {question}
    Search Results: {search_results}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def fact_validator(question, answer):
    """Validates the answer for factual correctness."""
    system_instruction = "You are a strict fact validator. Determine if the provided answer is factually correct based on your knowledge. If not, provide corrected answer."
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Is this answer factually correct? If not, provide a corrected answer.

    Example:
    Question: What is the capital of France?
    Answer: London
    Validation: INCORRECT. The capital of France is Paris.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def temporal_validator(question, answer):
    """Validates the answer for temporal consistency."""
    system_instruction = "You are an expert in temporal validation. Ensure the answer is temporally consistent with the question and the context. Provide feedback if dates don't align."
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Is this answer temporally consistent? If not, provide the reasoning and a corrected date if available.

    Example:
    Question: What year did World War II begin?
    Answer: 1930
    Temporal Validation: INCORRECT. World War II began in 1939.

    Question: {question}
    Answer: {answer}
    Temporal Validation:
    """
    return call_llm(prompt, system_instruction)

def source_reliability_validator(question, answer):
    """Validates the reliability of the source."""
    system_instruction = "You are an expert at source validation. Assess if the provided source is reliable and trustworthy."
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Assess the reliability of the source. State whether is it deemed reliable or not and why.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra Source: en.wikipedia.org/wiki/Canberra
    Source Validation: RELIABLE. Wikipedia is a generally reliable source of information.

    Question: {question}
    Answer: {answer}
    Source Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Solve questions using multiple search engines and layered validation."""
    search_results = simulate_search(question, 1)
    answer_with_source = extract_answer(question, search_results)

    fact_validation = fact_validator(question, answer_with_source)
    temporal_validation = temporal_validator(question, answer_with_source)
    source_validation = source_reliability_validator(question, answer_with_source)

    if "INCORRECT" in fact_validation:
        return fact_validation
    if "INCORRECT" in temporal_validation:
        return temporal_validation
    if "NOT RELIABLE" in source_validation:
        return source_validation
    else:
        return answer_with_source