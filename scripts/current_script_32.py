import os
import re
import math

# Hypothesis: Using a "Question Decomposition with Targeted Information Retrieval and a Knowledge Graph Validator" approach.
# This script introduces a decomposition approach, where a complex question is broken down into smaller questions, for each question, knowledge sources are queried to synthesize an answer. The knowledge graph is used to check if the answer is consistent.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def decompose_question(question):
    """Decompose a complex question into simpler questions."""
    system_instruction = "You are an expert question decomposer, breaking down questions into simpler, answerable parts."
    prompt = f"""
    Decompose the following complex question into simpler, answerable questions.

    Example 1:
    Complex Question: What is the capital of the country where the Great Barrier Reef is located, and what is the population of that capital?
    Decomposed Questions:
    1. Where is the Great Barrier Reef located?
    2. What is the capital of Australia?
    3. What is the population of Canberra?

    Complex Question: {question}
    Decomposed Questions:
    """
    return call_llm(prompt, system_instruction)

def retrieve_information(question):
    """Retrieve information from a simulated search engine."""
    system_instruction = "You are a search engine simulator providing factual and concise information. Be concise."
    prompt = f"""
    Simulate search results for the query: '{question}'.

    Example:
    Query: capital of Australia
    Search Results: Canberra is the capital of Australia.

    Query: {question}
    Search Results:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, retrieved_info):
    """Extract the answer from the retrieved information."""
    system_instruction = "You are an answer extraction expert, focusing on precision."
    prompt = f"""
    Extract the concise answer from the search results.

    Example:
    Question: What is the capital of Australia?
    Search Results: Canberra is the capital of Australia.
    Answer: Canberra

    Question: {question}
    Search Results: {retrieved_info}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def knowledge_graph_validator(question, answer):
    """Validates the answer against a simulated knowledge graph."""
    system_instruction = "You are a knowledge graph validator. Check if the answer is consistent with known facts."
    prompt = f"""
    Validate if the answer is consistent with a knowledge graph of facts.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Validation: VALID - Canberra is the capital of Australia.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Solve questions by decomposing, retrieving, extracting, and validating."""
    try:
        # 1. Decompose the question
        decomposed_questions = decompose_question(question)
        print(f"Decomposed questions: {decomposed_questions}") #debugging

        answers = []
        # 2. Process each sub-question
        for sub_question in decomposed_questions.split("\n"):
            if not sub_question.strip():
                continue  # Skip empty lines

            # 3. Retrieve Information
            retrieved_info = retrieve_information(sub_question)
            print(f"Retrieved info: {retrieved_info}") #debugging

            # 4. Extract Answer
            answer = extract_answer(sub_question, retrieved_info)
            print(f"Extracted answer: {answer}") #debugging

            # 5. Validate Answer
            validation_result = knowledge_graph_validator(sub_question, answer)
            print(f"Validation result: {validation_result}") #debugging

            if "VALID" not in validation_result:
                return "Could not be validated."
            answers.append(answer)

        # 6. Synthesize the final answer (simple concatenation for now)
        final_answer = ", ".join(answers)
        return final_answer

    except Exception as e:
        return f"Error: {str(e)}"