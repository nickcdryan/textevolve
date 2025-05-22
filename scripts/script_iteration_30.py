import os
import re
import math

# Hypothesis: Implementing a "Chain of Knowledge" approach with Iterative Fact Verification and Adaptive Source Selection
# This script introduces a "Chain of Knowledge" approach, where information is iteratively refined through a chain of LLM calls,
# focusing on Adaptive Source Selection. Source is selected based on previous reliability.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. """
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

def select_source(question, previous_sources=None):
    """Select the best source based on the question and past reliability."""
    system_instruction = "You are an expert source selector, choosing the most reliable information source."
    prompt = f"""
    Select the best source for answering the question, considering past source reliability.

    Example 1:
    Question: What is the capital of Australia?
    Past Sources: None
    Best Source: Wikipedia (General Knowledge, High Reliability)

    Example 2:
    Question: In what year was the praying mantis species Eremiaphila bifasciata described?
    Past Sources: Wikipedia (Reliability: High)
    Best Source: Zoological Record (Zoology, High Reliability)

    Question: {question}
    Past Sources: {previous_sources or "None"}
    Best Source:
    """
    return call_llm(prompt, system_instruction)

def retrieve_information(question, source):
    """Retrieve information from the selected source."""
    system_instruction = f"You are an information retriever, extracting relevant data from {source}."
    prompt = f"""
    Retrieve the most relevant information from {source} to answer the question.

    Example:
    Question: What is the capital of Australia?
    Source: Wikipedia
    Retrieved Information: Canberra is the capital of Australia.

    Question: {question}
    Source: {source}
    Retrieved Information:
    """
    return call_llm(prompt, system_instruction)

def verify_information(question, retrieved_info, source):
    """Verify the accuracy of the retrieved information."""
    system_instruction = "You are a fact verifier, ensuring the accuracy of information from {source}."
    prompt = f"""
    Verify if the information from {source} accurately answers the question.

    Example:
    Question: What is the capital of Australia?
    Retrieved Information: Canberra is the capital of Australia.
    Source: Wikipedia
    Verification: VALID - Canberra is indeed the capital of Australia.

    Question: {question}
    Retrieved Information: {retrieved_info}
    Source: {source}
    Verification:
    """
    verification_result = call_llm(prompt, system_instruction)
    return verification_result

def extract_answer(question, verified_info):
    """Extract a concise answer from the verified information."""
    system_instruction = "You are a concise answer extractor, focusing on precision."
    prompt = f"""
    Extract a concise answer from the verified information.

    Example:
    Question: What is the capital of Australia?
    Verified Information: VALID - Canberra is indeed the capital of Australia.
    Answer: Canberra

    Question: {question}
    Verified Information: {verified_info}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Solve questions using the Chain of Knowledge approach."""
    try:
        # Initialize variables
        previous_sources = None
        source = None

        # Select the best initial source
        source = select_source(question, previous_sources)
        print(f"Source: {source}")  # Debug print

        # Retrieve information from the selected source
        retrieved_info = retrieve_information(question, source)
        print(f"Retrieved info: {retrieved_info}") # Debug print

        # Verify the retrieved information
        verification_result = verify_information(question, retrieved_info, source)
        print(f"Verification: {verification_result}") # Debug print

        # Extract a concise answer
        answer = extract_answer(question, verification_result)
        print(f"Answer: {answer}") # Debug print

        # Return the answer
        return answer

    except Exception as e:
        return f"Error: {str(e)}"