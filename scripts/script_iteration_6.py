import os
import re
import json
import math # for react
from google import genai
from google.genai import types

# Combination of Iteration 2 (Validation Loop) and Iteration 4 (RAG with validated query)

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
    (Adapted from Iteration 4)
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question"

    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question. Be specific and target the precise information needed.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: "Ralph E. Oesper first name" exact

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: "Maharaj Kishan Bhan Padma Bhushan year"

        Example 3:
        Question: What day, month, and year was the first Sazae-san strip run by the Asahi Shimbun published?
        Search Query: "first Sazae-san strip published date Asahi Shimbun"

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction_query_gen)

        # Step 2: Simulate Retrieving Top Search Snippets - IMPORTANT: IN A REAL SYSTEM THIS WOULD BE SEARCH API
        search_snippets = call_llm(f"Provide top 3 search snippets for: {search_query}", "You are a helpful search engine providing realistic search results.")

        # Step 3: Validate Relevance of Search Snippets with Examples
        validation_prompt = f"""
        Determine if the following search snippets are relevant to answering the question. If they are, respond with "RELEVANT: [brief explanation]". If not, respond with "IRRELEVANT: [detailed explanation]".

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Snippets: Ralph Oesper was a professor...; His middle name was E...; There is no information on his first name.
        Validation: IRRELEVANT: The snippets don't reveal his first name.

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Snippets: Maharaj Kishan Bhan received the Padma Bhushan in 2013; He was a scientist; He worked in civil services.
        Validation: RELEVANT: Snippets contain MKB and the year he received the award.

        Example 3:
        Question: What day, month, and year was the first Sazae-san strip run by the Asahi Shimbun published?
        Search Snippets: The first Sazae-san strip was published on November 30, 1949.
        Validation: RELEVANT: Snippet provides the full date of publication.

        Question: {question}
        Search Snippets: {search_snippets}
        Validation:
        """
        validation_result = call_llm(validation_prompt, system_instruction_search_validator)

        if "RELEVANT:" in validation_result:
            return search_query, search_snippets  # Return both the search query and relevant snippets
        else:
            print(f"Attempt {attempt + 1}: Search snippets deemed irrelevant. Trying again...")

    return None, None  # Return None if no relevant context is found

def generate_answer_with_snippets(question, search_snippets):
    """
    Generates an answer using the validated search snippets. (Adapted from Iteration 4)
    """
    system_instruction = "You are an expert at answering questions given relevant search snippets. Provide the most accurate and complete answer possible, extracting key details."

    answer_prompt = f"""
    Answer the question using ONLY the information present in the search snippets. Extract as much detail as possible. If the answer is not found, clearly state "Answer not found in snippets".

    Example 1:
    Question: What was the first name of Ralph E. Oesper?
    Search Snippets: No results found.
    Answer: Answer not found in snippets.

    Example 2:
    Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
    Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
    Answer: 2013

    Example 3:
    Question: What day, month, and year was the first Sazae-san strip run by the Asahi Shimbun published?
    Search Snippets: The first Sazae-san strip by Sazae-san was published by the Asahi Shimbun on November 30, 1949.
    Answer: November 30, 1949

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def solve_with_validation_loop(problem, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop.
    This is primarily adapted from Iteration 2, but using RAG from Iteration 4"""

    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements, ensuring factual correctness and completeness."

    # RAG process from Iteration 4
    search_query, search_snippets = generate_query_and_validate(problem)

    if not search_query or not search_snippets:
        return "Answer not found after multiple attempts."

    initial_solution = generate_answer_with_snippets(problem, search_snippets)

    # Validation loop (from Iteration 2, but adapted to RAG output)
    solution = initial_solution
    for attempt in range(max_attempts):
        # Validate the current solution
        validation_prompt = f"""
        Carefully validate if this solution correctly answers all aspects of the problem, is consistent with the search snippets, and is factually correct.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues (e.g. inaccuracies, missing information, or inconsistency with the search snippets), respond with "INVALID: [detailed explanation of issues]".

        Example 1:
        Problem: What is the capital of France?
        Proposed Solution: Paris
        Validation: VALID: The capital of France is indeed Paris.

        Example 2:
        Problem: Who painted the Mona Lisa?
        Proposed Solution: Leonardo da Vinci
        Validation: VALID: The Mona Lisa was painted by Leonardo da Vinci.

        Example 3:
        Problem: What year did World War II begin?
        Proposed Solution: 1940
        Validation: INVALID: World War II began in 1939, not 1940.

        Problem: {problem}
        Proposed Solution: {solution}
        Search Snippets: {search_snippets}
        Validation:
        """

        validation_result = call_llm(validation_prompt, system_instruction_validator)

        # Check if solution is valid
        if validation_result.startswith("VALID:"):
            return solution

        # If invalid, refine the solution
        refined_prompt = f"""
        Your previous solution to this problem has some issues that need to be addressed. Ensure that your solution uses ONLY information extracted from the search snippets provided, and that it is factually correct.

        Problem: {problem}
        Proposed Solution: {solution}
        Search Snippets: {search_snippets}
        Validation feedback: {validation_result}

        Example of a corrected solution based on validation feedback:

        Problem: When did the Titanic sink?
        Proposed Solution: April 1912
        Search Snippets: The Titanic sank on April 15, 1912.
        Validation Feedback: INVALID: Missing the specific day that the Titanic sank which the search snippets contained.

        Corrected Solution: April 15, 1912

        Generate an improved solution. Use the search snippets to extract the correct answer and all required details. If the snippets don't contain the answer, state "Answer not found in snippets."
        """
        solution = generate_answer_with_snippets(problem, search_snippets) #Force resynthesis from snippets
        if "Answer not found in snippets" in solution: #Break if snippets truly don't have the answer
             return solution

    return solution # If validation loop fails, return the last solution

def main(question):
    """
    Main function that orchestrates the solution process using solve_with_validation_loop.
    This function now incorporates RAG and an iterative validation loop for enhanced accuracy.
    """
    answer = solve_with_validation_loop(question)
    return answer