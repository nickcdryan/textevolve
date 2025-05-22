import os
import re
import math

# Enhanced Fact-Checking with Multiple Simulated Search Engines and Improved Validation.
# This version builds upon the previous best (iteration 22) by adding examples to prompts,
# improving the simulate_search function, and adding a new answer verification mechanism.

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

def simulate_search(query, engine_id):
    """Simulate different search engines with more diverse results."""
    system_instruction = f"You are a simulated search engine with ID {engine_id} providing factual information with a specific bias. Be concise but credible."
    prompt = f"""
    Simulate diverse search results for the query: '{query}'.

    Example 1 (Engine ID: 1, general knowledge):
    Query: capital of France
    Search Results: Paris is the capital and most populous city of France.

    Example 2 (Engine ID: 2, historical focus):
    Query: capital of France
    Search Results: Historically, Paris has been the center of French power and culture.

    Example 3 (Engine ID: 3, slightly incorrect):
    Query: capital of France
    Search Results: Some might mistakenly believe Lyon is the capital, but it's Paris.

    Query: {query}
    Search Results:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, search_results):
    """Extract potential answers from search results with an example."""
    system_instruction = "You are an answer extraction expert, focusing on precision. Extract the concise answer only."
    prompt = f"""
    Extract the concise answer to the question from the search results.

    Example:
    Question: What is the capital of Australia?
    Search Results: Canberra is the capital of Australia.
    Answer: Canberra

    Question: {question}
    Search Results: {search_results}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def reconcile_answers(question, answers):
    """Reconcile answers from different engines with an example."""
    system_instruction = "You are an expert at reconciling conflicting answers. Determine the most accurate answer. Prioritize factual correctness."
    all_answers = "\n".join([f"Engine {i+1}: {answer}" for i, answer in enumerate(answers)])
    prompt = f"""
    Reconcile these answers from different sources to answer the question.

    Example:
    Question: What is the capital of Australia?
    Engine 1: Canberra
    Engine 2: Canberra is the capital city.
    Reconciled Answer: Canberra

    Question: {question}
    {all_answers}
    Reconciled Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, reconciled_answer):
    """Validate the reconciled answer. Returns VALID or INVALID."""
    system_instruction = "You are a strict validator, focusing on factual correctness. Determine whether answer is factually correct or not. If so, label it VALID with reasoning."""
    prompt = f"""
    Validate if the reconciled answer is correct for the question. Explain your reasoning.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Validation: VALID - Canberra is the capital of Australia.

    Question: {question}
    Answer: {reconciled_answer}
    Validation:
    """
    validation_result = call_llm(prompt, system_instruction)
    return validation_result

def main(question):
    """Solve questions using multiple search engines and answer reconciliation."""
    num_engines = 3
    answers = []

    # Simulate search with multiple engines
    for i in range(num_engines):
        search_results = simulate_search(question, i+1)
        answer = extract_answer(question, search_results)
        answers.append(answer)

    # Reconcile answers
    reconciled_answer = reconcile_answers(question, answers)

    # Validate answer
    validation_result = validate_answer(question, reconciled_answer)

    if "VALID" in validation_result:
        return reconciled_answer
    else:
        return "Could not be validated."