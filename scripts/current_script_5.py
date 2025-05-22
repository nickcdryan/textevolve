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

    This function incorporates elements from Iteration 4 (RAG approach)
    and enhances the query generation and validation steps.
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question. Focus on factual recall and completeness."

    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples (From Iteration 4)
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question. Focus on generating queries to help factually answer the questions.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: Ralph E. Oesper first name

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: Maharaj Kishan Bhan Padma Bhushan year

        Example 3:
        Question: Which of the three Olympic fencing weapons was the last one to transition to using electrical equipment?
        Search Query: Olympic fencing weapons electrical equipment transition

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction_query_gen)

        # Step 2: Simulate Retrieving Top Search Snippets
        search_snippets = call_llm(f"Provide top 3 search snippets for: {search_query}", "You are a helpful search engine providing realistic search results. Focus on factual and complete information")

        # Step 3: Validate Relevance of Search Snippets with Examples (From Iteration 4, enhanced)
        validation_prompt = f"""
        Determine if the following search snippets are relevant to answering the question. If they are, respond with "RELEVANT: [brief explanation]". If not, respond with "IRRELEVANT: [detailed explanation]". Prioritize factual recall.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Snippets: Ralph Oesper was a professor...; His middle name was E...; There is no information on his first name.
        Validation: IRRELEVANT: The snippets don't reveal his first name or a direct answer.

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Snippets: Maharaj Kishan Bhan received the Padma Bhushan in 2013; He was a scientist; He worked in civil services.
        Validation: RELEVANT: Snippets contain MKB and the year he received the award providing a direct answer.

        Example 3:
        Question: In which of the three Olympic fencing weapons was the last one to transition to using electrical equipment?
        Search Snippets: The sabre was the last of the three weapons to be electrified.
        Validation: RELEVANT: Provides answer as to which weapon and the electrical equipment transition

        Question: {question}
        Search Snippets: {search_snippets}
        Validation:
        """
        validation_result = call_llm(validation_prompt, system_instruction_search_validator)

        if "RELEVANT:" in validation_result:
            return search_query, search_snippets
        else:
            print(f"Attempt {attempt + 1}: Search snippets deemed irrelevant. Trying again...")

    return None, None

def generate_answer_with_snippets(question, search_snippets):
    """
    Generates an answer using the validated search snippets.

    This function is from Iteration 4, but adds emphasis to extract specific
    facts, particularly years and names.
    """
    system_instruction = "You are an expert at answering questions given relevant search snippets. Focus on extracting specific facts like names and years."

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
    Question: Which of the three Olympic fencing weapons was the last one to transition to using electrical equipment?
    Search Snippets: The sabre was the last of the three weapons to be electrified.
    Answer: Sabre

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def solve_with_validation_loop(question, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop.
    This incorporates the validation loop approach from Iteration 2
    to refine the answer generated from the RAG approach of Iteration 4.
    """
    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions. Focus on factual accuracy and completeness."
    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements, ensuring factual correctness and completeness. Focus on directly answering the question."

    # Initial solution generation using RAG
    search_query, search_snippets = generate_query_and_validate(question)
    if search_query and search_snippets:
        solution = generate_answer_with_snippets(question, search_snippets)
    else:
        solution = "Answer not found."

    # Validation loop (From Iteration 2)
    for attempt in range(max_attempts):
        # Validate the current solution
        validation_prompt = f"""
        Carefully validate if this solution correctly addresses all aspects of the problem. Ensure factual correctness and completeness. The response should directly answer the question.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues, respond with "INVALID: [detailed explanation of issues, including specific factual errors or omissions]".

        Example 1:
        Problem: What is the capital of France?
        Solution: Paris
        Validation: VALID: The capital of France is indeed Paris. Provides a factually correct response.

        Example 2:
        Problem: Who painted the Mona Lisa?
        Solution: Leonardo DaVinci
        Validation: VALID: The Mona Lisa was painted by Leonardo da Vinci and this directly answers the question.

        Example 3:
        Problem: What year did World War II begin?
        Solution: 1940
        Validation: INVALID: World War II began in 1939, not 1940. Contains factually incorrect information and should be revised.

        Problem:
        {question}

        Proposed Solution:
        {solution}
        """

        validation_result = call_llm(validation_prompt, system_instruction_validator)

        # Check if solution is valid
        if "VALID:" in validation_result:
            return solution

        # If invalid, refine the solution. Attempt to regenerate the answer given the error,
        # as the initial search query may have been inadequate

        search_query, search_snippets = generate_query_and_validate(question)  # Re-run to generate new snippets
        if search_query and search_snippets:
            solution = generate_answer_with_snippets(question, search_snippets)
        else:
            solution = "Answer not found."

    return solution # Returns best attempt by end
def main(question):
    """
    Main function that orchestrates the solution process using solve_with_validation_loop.
    """
    answer = solve_with_validation_loop(question)
    return answer