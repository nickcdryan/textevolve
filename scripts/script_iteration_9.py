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
    Combines techniques from Iteration 4.
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question"

    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: Ralph E. Oesper first name

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: Maharaj Kishan Bhan Padma Bhushan year

        Example 3:
        Question: What hospital did Communist politician Georgi Dimitrov die in 1949?
        Search Query: Georgi Dimitrov death hospital 1949

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
        Validation: RELEVANT: Snippets contain MKB and the year he received the award

        Example 3:
        Question: What hospital did Communist politician Georgi Dimitrov die in 1949?
        Search Snippets: Georgi Dimitrov died in the Barvikha sanatorium in 1949; He was a Bulgarian politician.
        Validation: RELEVANT: Snippets mention Dimitrov, death and the hospital.

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
    Combines techniques from Iteration 4, with multi-example prompting.
    """
    system_instruction = "You are an expert at answering question given relevant search snippets"
    # Now we leverage the search snippets to answer the question directly
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
    Question: What hospital did Communist politician Georgi Dimitrov die in 1949?
    Search Snippets: Georgi Dimitrov died in the Barvikha sanatorium in 1949; He was a Bulgarian politician.
    Answer: Barvikha sanatorium

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def solve_with_validation_loop(question, max_attempts=3):
    """Solve a problem with iterative refinement through validation feedback loop.
    Combines techniques from Iteration 2 and Iteration 4. Includes multi-example prompting for improved accuracy.
    """
    system_instruction_solver = "You are an expert problem solver who creates detailed, correct solutions. Focus on factual accuracy and completeness."
    system_instruction_validator = "You are a critical validator who carefully checks solutions against all requirements, ensuring factual correctness and completeness."

    # Initial solution generation - Enhanced with multi-example prompting
    solution_prompt = f"""
    Provide a detailed solution to this problem. Be thorough and ensure you address all requirements. Focus on factually accurate and complete answers.

    Example 1:
    Problem: What is the name of the university where Ana Figueroa, a political activist and government official, studies and graduates from?
    Solution: University of Chile

    Example 2:
    Problem: Which genus was the ruby-throated bulbul moved to from *Turdus* before finally being classified in the genus *Rubigula*?
    Solution: Genus Pycnonotus

    Example 3:
    Problem: In what year did Etta Cone last visit Europe?
    Solution: 1938

    Problem:
    {question}
    """

    # Modified from Iteration 2: Uses RAG to get the initial solution.
    search_query, search_snippets = generate_query_and_validate(question)

    if search_query and search_snippets:
        solution = generate_answer_with_snippets(question, search_snippets)
    else:
        solution = "Answer not found."


    # Validation loop from Iteration 2
    for attempt in range(max_attempts):
        # Validate the current solution - Enhanced with specific validation examples
        validation_prompt = f"""
        Carefully validate if this solution correctly addresses all aspects of the problem. Ensure factual correctness and completeness.
        If the solution is valid, respond with "VALID: [brief reason]".
        If the solution has any issues, respond with "INVALID: [detailed explanation of issues, including specific factual errors or omissions]".

        Example 1:
        Problem: What is the capital of France?
        Solution: Paris
        Validation: VALID: The capital of France is indeed Paris.

        Example 2:
        Problem: Who painted the Mona Lisa?
        Solution: Leonardo DaVinci
        Validation: VALID: The Mona Lisa was painted by Leonardo da Vinci.

        Example 3:
        Problem: What year did World War II begin?
        Solution: 1940
        Validation: INVALID: World War II began in 1939, not 1940.

        Problem:
        {question}

        Proposed Solution:
        {solution}
        """

        validation_result = call_llm(validation_prompt, system_instruction_validator)

        # Check if solution is valid
        if validation_result.startswith("VALID:"):
            return solution

        # If invalid, refine the solution - Provides multi-example based feedback to ensure robust refinement
        refined_prompt = f"""
        Your previous solution to this problem has some issues that need to be addressed. Ensure that you only use information from the original problem in your response, and ensure that the response is factually correct and as complete as possible.

        Problem:
        {question}

        Your previous solution:
        {solution}

        Validation feedback:
        {validation_result}

        Example of a corrected solution based on validation feedback:

        Problem: When did the Titanic sink?
        Your previous solution: April 1912
        Validation Feedback: INVALID: The Titanic sank on April 15, 1912, include the day.

        Corrected Solution: April 15, 1912

        Please provide a completely revised solution that addresses all the issues mentioned. Be as factual as possible. Do not attempt to create new information that is not present in the original response.
        """

        # Modified from Iteration 2: Now, use the search snippets to guide refinement:
        search_query, search_snippets = generate_query_and_validate(question)

        if search_query and search_snippets:
            solution = generate_answer_with_snippets(question, search_snippets)
        else:
            solution = call_llm(refined_prompt, system_instruction_solver) # Fallback to direct LLM if search fails.

    return solution

def main(question):
    """
    Main function that orchestrates the solution process using solve_with_validation_loop.
    Combines elements from Iteration 2 and Iteration 4. This is a hybrid approach
    """
    answer = solve_with_validation_loop(question)
    return answer