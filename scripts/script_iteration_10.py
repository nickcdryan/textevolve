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
    Generates a search query from a question and validates its effectiveness by assessing
    if the top search snippets contain key entities and relationships needed to answer the question.
    Returns both the generated query and top search snippets.
    This function integrates aspects from Iteration 4 (RAG) with improved prompt engineering.
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question"
    # Hypothesis: By generating and validating the query BEFORE retrieving the information, we can significantly improve the information retrieval and hallucination problems that are causing the pipeline to fail
    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples - Adapted from Iteration 4
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question. Consider the type of question and what the answer will be when generating your query.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: Ralph E. Oesper first name

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: Maharaj Kishan Bhan Padma Bhushan year civil services

        Example 3:
        Question: On what date (day/month/year) was Makhdum Khusro Bakhtyar (Pakistani politician) inducted into the Federal Cabinet of Prime Minister Shaukat Aziz?
        Search Query: Makhdum Khusro Bakhtyar inducted into Federal Cabinet date

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction_query_gen)
        # Step 2: Simulate Retrieving Top Search Snippets - IMPORTANT: IN A REAL SYSTEM THIS WOULD BE SEARCH API
        search_snippets = call_llm(f"Provide top 3 search snippets for: {search_query}", "You are a helpful search engine providing realistic search results.")

        # Step 3: Validate Relevance of Search Snippets with Examples - Adapted from Iteration 4, modified to be more strict
        validation_prompt = f"""
        Determine if the following search snippets are highly relevant to answering the question. If they are, respond with "RELEVANT: [brief explanation]". If not, respond with "IRRELEVANT: [detailed explanation]". The snippets must contain the answer to the question DIRECTLY or lead to it.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Snippets: Ralph Oesper was a professor...; His middle name was E...; There is no information on his first name.
        Validation: IRRELEVANT: The snippets don't reveal his first name.

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
        Validation: RELEVANT: Snippets contain Maharaj Kishan Bhan and the year 2013.

        Example 3:
        Question: On what date (day/month/year) was Makhdum Khusro Bakhtyar (Pakistani politician) inducted into the Federal Cabinet of Prime Minister Shaukat Aziz?
        Search Snippets: Makhdum Khusro Bakhtyar was inducted into the cabinet on 4 September 2004
        Validation: RELEVANT: The snippet states the date MKB was inducted into the cabinet.

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
    Generates an answer using the validated search snippets, ensuring that the answer
    is directly supported by the information in the snippets.
    This function takes aspects from Iteration 4 (RAG).
    """
    system_instruction = "You are an expert at answering question given relevant search snippets. You must only answer based on information contained in the snippets."
    # Now we leverage the search snippets to answer the question directly
    answer_prompt = f"""
    Answer the question using ONLY the information present in the search snippets. Provide a concise answer.

    Example 1:
    Question: What was the first name of Ralph E. Oesper?
    Search Snippets: No results found.
    Answer: Answer not found.

    Example 2:
    Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
    Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
    Answer: 2013

    Example 3:
    Question: On what date (day/month/year) was Makhdum Khusro Bakhtyar (Pakistani politician) inducted into the Federal Cabinet of Prime Minister Shaukat Aziz?
    Search Snippets: Makhdum Khusro Bakhtyar was inducted into the cabinet on 4 September 2004
    Answer: 4 September 2004

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def solve_with_validation_loop(question, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop.
    This function is based on Iteration 2, using the validation loop pattern.
    It's combined with RAG from Iteration 4 for stronger information retrieval."""
    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions. Focus on factual accuracy and completeness. Ensure your answers only come from search snippets."
    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements, ensuring factual correctness and completeness. Verify the answer comes from search snippets"

    # Initial solution generation - Now uses RAG from Iteration 4
    search_query, search_snippets = generate_query_and_validate(question)
    if search_query and search_snippets:
        solution = generate_answer_with_snippets(question, search_snippets)
    else:
        return "Answer not found."

    # Validation loop - From Iteration 2
    for attempt in range(max_attempts):
        # Validate the current solution - Enhanced with specific validation examples
        validation_prompt = f"""
        Carefully validate if this solution correctly addresses all aspects of the problem. Ensure factual correctness and completeness, and that the answer comes only from the provided search snippets.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues, respond with "INVALID: [detailed explanation of issues, including specific factual errors or omissions or a source outside of the snippets]".

        Example 1:
        Question: What is the capital of France?
        Search Snippets: Paris is the capital of France.
        Solution: Paris
        Validation: VALID: The capital of France is indeed Paris, and the snippet confirms this.

        Example 2:
        Question: Who painted the Mona Lisa?
        Search Snippets: Leonardo da Vinci painted the Mona Lisa.
        Solution: Leonardo DaVinci
        Validation: VALID: The Mona Lisa was painted by Leonardo da Vinci, as confirmed in the search snippet.

        Example 3:
        Question: What year did World War II begin?
        Search Snippets: The Second World War started in 1939.
        Solution: 1940
        Validation: INVALID: World War II began in 1939, not 1940, as stated in the search snippet.

        Question:
        {question}

        Proposed Solution:
        {solution}

        Search Snippets:
        {search_snippets}
        """

        validation_result = call_llm(validation_prompt, system_instruction_validator)

        # Check if solution is valid
        if validation_result.startswith("VALID:"):
            return solution

        # If invalid, try a different approach (query)
        else:
            print(f"Validation failed. Retrying with new query...")
            search_query, search_snippets = generate_query_and_validate(question)  # Generate a new query
            if search_query and search_snippets:
                solution = generate_answer_with_snippets(question, search_snippets)  # Generate answer with new snippets
            else:
                return "Answer not found."  # If still no snippets, give up

    return solution # Return the best effort after max attempts

def main(question):
    """
    Main function that orchestrates the solution process using solve_with_validation_loop.
    This function now incorporates the iterative validation loop for enhanced accuracy.
    This is a hybrid approach combining elements from Iteration 2 and Iteration 4.
    """
    answer = solve_with_validation_loop(question)
    return answer