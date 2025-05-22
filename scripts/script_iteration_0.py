import os
import re
import math

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    """Solve factual questions using iterative retrieval and refinement with detailed validation."""

    # Hypothesis: Integrating web search *directly* into the LLM prompting, rather than as a separate action, will improve accuracy.

    # Step 1: Generate initial search query based on the question (with examples)
    search_query_prompt = f"""
    Given a factual question, generate a concise and effective search query that will retrieve relevant information from the web.

    Example 1:
    Question: What was the first name of the wife of the American chemist Ralph E. Oesper?
    Search Query: Ralph E. Oesper wife's name

    Example 2:
    Question: Who formed the Dubai-based band Sho? in June 2009?
    Search Query: Dubai band Sho formed June 2009

    Question: {question}
    Search Query:
    """
    search_query = call_llm(search_query_prompt, "You are a search query generator.")

    # Step 2: Embed search query and retrieve information (simulated)
    retrieved_info = f"Simulated web search results for: {search_query}. Placeholder for real search functionality."  # Replace with actual search API call

    # Step 3: Refine and extract answer from retrieved information (with examples and validation)
    answer_extraction_prompt = f"""
    Given a question and relevant information, extract the answer and provide a confidence score (1-10).

    Example 1:
    Question: What was the first name of the wife of the American chemist Ralph E. Oesper?
    Relevant Information: Helen Oesper was the wife of Ralph E. Oesper.

    Let's think step by step.
    The question is about the first name of Ralph E. Oesper's wife.
    The relevant information clearly states Helen Oesper was his wife.
    So, the answer is Helen.
    Confidence Score: 10

    Answer: Helen
    Confidence Score: 10

    Example 2:
    Question: In the series "El guardián invisible," who portrays the character Alfonso Álvarez de Toledo?
    Relevant Information: Ramón Barea played Alfonso Álvarez de Toledo in "El guardián invisible".

    Let's think step by step.
    The question is about who portrays the character Alfonso Álvarez de Toledo in "El guardián invisible."
    The relevant information clearly states Ramón Barea played the character.
    So, the answer is Ramón Barea.
    Confidence Score: 10

    Answer: Ramón Barea
    Confidence Score: 10
    
    Question: {question}
    Relevant Information: {retrieved_info}

    Let's think step by step.
    """

    # Step 4: Make an independent validation call.
    verification_prompt = f"""
    Question: {question}
    Retrieved Information: {retrieved_info}
    Extracted answer: {extracted_answer}

    The question is:
    {question}
    Given the problem statement and the relevant information, validate the answer, step by step.
    If the answer is incorrect, return 'Incorrect'. If the answer is correct, return 'Correct'.
    """

    validation = call_llm(prompt=verification_prompt, system_instruction="You are an expert at validating answers based on a question and provided information. Return a plain language answer of 'Correct' or 'Incorrect'.")

    # If the validation is correct, return the answer. If not, respond that the answer cannot be validated.
    if validation == 'Correct':
        answer_extraction_response = call_llm(answer_extraction_prompt, "You are a question answering expert.")
        extracted_answer = answer_extraction_response.split("Answer:")[1].split("Confidence Score:")[0].strip()
    elif validation == 'Incorrect':
        return "Could not be validated."
    else:
        return "Could not be validated."

    return extracted_answer