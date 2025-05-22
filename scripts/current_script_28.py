import os
import re
import math

# Hypothesis: Implementing a two-stage retrieval and focused summarization approach with self-reflection.
# The first stage retrieves broad information, then the second stage summarizes with respect to the question.
# A self-reflection mechanism validates the answer.

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

def initial_info_retrieval(question):
    """Retrieve initial information related to the question."""
    system_instruction = "You are an information retrieval expert. Find relevant background information."
    prompt = f"""
    Provide background information that could be useful in answering the question.

    Example:
    Question: What is the capital of Australia?
    Background Information: Australia is a country in the southern hemisphere. Its largest city is Sydney, but its capital is Canberra.

    Question: {question}
    Background Information:
    """
    return call_llm(prompt, system_instruction)

def focused_summarization(question, background_info):
    """Summarize the background information, focusing on the question."""
    system_instruction = "You are a summarization expert, focusing on answering the given question."
    prompt = f"""
    Summarize the background information with respect to the specific question.

    Example:
    Question: What is the capital of Australia?
    Background Information: Australia is a country in the southern hemisphere. Its largest city is Sydney, but its capital is Canberra.
    Focused Summary: The capital of Australia is Canberra.

    Question: {question}
    Background Information: {background_info}
    Focused Summary:
    """
    return call_llm(prompt, system_instruction)

def self_reflection_validation(question, summarized_info):
    """Validate the summarized information and answer the question."""
    system_instruction = "You are a validation expert, verifying the accuracy of the information and answering the question."
    prompt = f"""
    Validate the summarized information and answer the question concisely. Determine if it answers the question accurately, and provide a concise answer. If validation fails respond with 'Could not be validated.'

    Example:
    Question: What is the capital of Australia?
    Summarized Information: The capital of Australia is Canberra.
    Validation and Answer: Canberra

    Question: {question}
    Summarized Information: {summarized_info}
    Validation and Answer:
    """
    validation_result = call_llm(prompt, system_instruction)
    return validation_result

def main(question):
    """Solve questions using two-stage retrieval and self-reflection."""
    try:
        # Initial Information Retrieval
        background_info = initial_info_retrieval(question)

        # Focused Summarization
        summarized_info = focused_summarization(question, background_info)

        # Self-Reflection Validation and Answer
        validation_result = self_reflection_validation(question, summarized_info)

        return validation_result

    except Exception as e:
        return f"Error: {str(e)}"