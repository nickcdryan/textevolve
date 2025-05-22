import os
import re
import json
import math # for react
from google import genai
from google.genai import types

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

def generate_query_and_validate(question, max_attempts=3):
    """
    Generates a search query from a question and validates its effectiveness.
    Returns both the generated query and top search snippets.
    Technique derived from Iteration 4.
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question"

    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples - Multi-example prompting
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: Ralph E. Oesper first name

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: Maharaj Kishan Bhan Padma Bhushan year

        Example 3:
        Question: What year did Australian politician William Lawrence Morrison graduate from the University of Sydney?
        Search Query: William Lawrence Morrison graduation year University of Sydney

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction_query_gen)

        # Step 2: Simulate Retrieving Top Search Snippets
        search_snippets = call_llm(f"Provide top 3 search snippets for: {search_query}", "You are a helpful search engine providing realistic search results.")

        # Step 3: Validate Relevance of Search Snippets with Examples - Multi-example prompting
        validation_prompt = f"""
        Determine if the following search snippets are relevant to answering the question. If they are, respond with "RELEVANT: [brief explanation]". If not, respond with "IRRELEVANT: [detailed explanation]".

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Snippets: Ralph Oesper was a professor...; His middle name was E...; There is no information on his first name.
        Validation: IRRELEVANT: The snippets don't reveal his first name.

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
        Validation: RELEVANT: Snippets contain MKB and the year he received the award

        Example 3:
        Question: What year did Australian politician William Lawrence Morrison graduate from the University of Sydney?
        Search Snippets: William Lawrence Morrison graduated from the University of Sydney in 1949...
        Validation: RELEVANT: The snippets contain the politician, university and graduation year.

        Question: {question}
        Search Snippets: {search_snippets}
        Validation:
        """
        validation_result = call_llm(validation_prompt, system_instruction_search_validator)

        if "RELEVANT:" in validation_result:
            return search_query, search_snippets # Return both the search query and relevant context
        else:
            print(f"Attempt {attempt + 1}: Search snippets deemed irrelevant. Trying again...")

    return None, None  # Return None if no relevant context is found

def generate_answer_with_snippets(question, search_snippets):
    """
    Generates an answer using the validated search snippets.
    Technique derived from Iteration 4.
    """
    system_instruction = "You are an expert at answering questions given relevant search snippets"
    # Now we leverage the search snippets to answer the question directly - Multi-example prompting
    answer_prompt = f"""
    Answer the question using ONLY the information present in the search snippets.

    Example 1:
    Question: What was the first name of Ralph E. Oesper?
    Search Snippets: No results found.
    Answer: Answer not found.

    Example 2:
    Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
    Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
    Answer: 2013

    Example 3:
    Question: What year did Australian politician William Lawrence Morrison graduate from the University of Sydney?
    Search Snippets: William Lawrence Morrison graduated from the University of Sydney in 1949. He went on to become a politician...
    Answer: 1949

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def verify_answer(question, answer):
    """Verifies the answer against the original question to ensure relevance and accuracy.
    Technique derived from Iteration 7."""
    system_instruction = "You are a critical validator who checks if an answer is factually correct and relevant to the question."
    prompt = f"""
    Verify if the following answer accurately and completely answers the question. Respond with VALID or INVALID, followed by a brief explanation.

    Example 1:
    Question: What is the capital of France?
    Answer: Paris
    Verification: VALID: Paris is indeed the capital of France.

    Example 2:
    Question: In what year did World War II begin?
    Answer: 1940
    Verification: INVALID: World War II began in 1939.

    Example 3:
    Question: What is the apparent visual magnitude of Gliese 146?
    Answer: 8.64
    Verification: VALID: 8.64 is the apparent visual magnitude of Gliese 146.

    Question: {question}
    Answer: {answer}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def solve_with_rag_and_verification(question, max_attempts=3):
    """Combines RAG (from Iteration 4) with answer verification (from Iteration 7)"""
    search_query, search_snippets = generate_query_and_validate(question)

    if search_query and search_snippets:
        answer = generate_answer_with_snippets(question, search_snippets)
        verification_result = verify_answer(question, answer)

        if "VALID" in verification_result:
            return answer
        else:
             # Attempt to refine the answer if the initial answer is invalid.
            refined_answer = generate_answer_with_snippets(question, search_snippets)
            refined_verification_result = verify_answer(question, refined_answer)
            if "VALID" in refined_verification_result:
                return refined_answer
            else:
                return "Could not find the answer."
    else:
        return "Could not find the answer."

def main(question):
    """
    Main function to orchestrate the solution process. This function combines techniques from
    iterations 4 and 7.
    """
    answer = solve_with_rag_and_verification(question)
    return answer