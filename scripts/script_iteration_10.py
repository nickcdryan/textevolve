import os
import re
import math

def main(question):
    """Schedules meetings using a new LLM-driven approach with explicit reasoning and staged solution building.

    HYPOTHESIS: By explicitly prompting the LLM to state the problem, constraints,
    propose multiple candidate solutions, and then select the best one, we can
    improve the quality and reliability of the generated meeting time.
    It also includes a validation step to determine which components of the solution pipeline are working successfully.

    This is a fundamentally different approach that focuses on a structured thought process
    and staged solution building, rather than simply extracting and proposing.
    """
    try:
        # 1. Analyze the problem and extract constraints
        analysis = analyze_problem(question)
        if "Error" in analysis:
            return analysis

        # 2. Generate candidate solutions with reasoning
        candidates = generate_candidate_solutions(analysis, question)
        if "Error" in candidates:
            return candidates

        # 3. Select the best candidate
        best_solution = select_best_solution(candidates, question)
        if "Error" in best_solution:
            return best_solution

        # 4. Verify proposed time (new verification for debugging)
        verified_solution = verify_proposed_time(best_solution, question)
        if "Error" in verified_solution:
            return verified_solution

        return best_solution

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def analyze_problem(question):
    """Analyzes the problem and extracts key constraints."""
    system_instruction = "You are an expert at analyzing scheduling problems."
    prompt = f"""
    You are an expert at analyzing scheduling problems. Your goal is to extract information.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Analysis:
    Problem: Schedule a 30-minute meeting for John and Mary on Monday.
    Constraints:
    - John is busy from 9:00-10:00 on Monday.
    - Mary is busy from 11:00-12:00 on Monday.

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Analysis:
    Problem: Schedule a 1-hour meeting for Alice, Bob, and Charlie on Tuesday and Wednesday.
    Constraints:
    - Alice is busy from 14:00-15:00 on Tuesday.
    - Bob is busy from 10:00-11:00 on Wednesday.

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def generate_candidate_solutions(analysis, question):
    """Generates multiple candidate solutions with explicit reasoning."""
    system_instruction = "You are an expert at generating candidate scheduling solutions."
    prompt = f"""
    You are an expert at generating candidate scheduling solutions.

    Analysis: {analysis}

    Question: {question}

    Example 1:
    Analysis:
    Problem: Schedule a 30-minute meeting for John and Mary on Monday.
    Constraints:
    - John is busy from 9:00-10:00 on Monday.
    - Mary is busy from 11:00-12:00 on Monday.
    Candidate Solutions:
    - Monday, 10:00-10:30 (John is free, Mary is free)
    - Monday, 12:00-12:30 (John is free, Mary is free)
    Reasoning: Considering the constraints, these two times work for both participants.

     Example 2:
    Analysis:
    Problem: Schedule a 1-hour meeting for Alice, Bob, and Charlie on Tuesday and Wednesday.
    Constraints:
    - Alice is busy from 14:00-15:00 on Tuesday.
    - Bob is busy from 10:00-11:00 on Wednesday.
    Candidate Solutions:
    - Tuesday, 10:00-11:00 (Alice, Bob, Charlie are free)
    - Wednesday, 11:00-12:00 (Alice, Bob, Charlie are free)
    Reasoning: These two times work for all participants.

    Question: {question}
    Candidate Solutions:
    """
    return call_llm(prompt, system_instruction)

def select_best_solution(candidates, question):
    """Selects the best candidate solution based on the generated reasoning."""
    system_instruction = "You are an expert at selecting the best scheduling solution."
    prompt = f"""
    You are an expert at selecting the best scheduling solution.

    Candidates: {candidates}

    Question: {question}

    Example 1:
    Candidates:
    - Monday, 10:00-10:30 (John is free, Mary is free)
    - Monday, 12:00-12:30 (John is free, Mary is free)
    Reasoning: Considering the constraints, the earliest time is best
    Best Solution: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Candidates:
    - Tuesday, 10:00-11:00 (Alice, Bob, Charlie are free)
    - Wednesday, 11:00-12:00 (Alice, Bob, Charlie are free)
    Reasoning: Considering the constraints, the earliest day is best
    Best Solution: Here is the proposed time: Tuesday, 10:00-11:00

    Question: {question}
    Best Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_proposed_time(proposed_time, question):
    """Verifies if the proposed time is valid"""
    system_instruction = "You are an expert at verifying meeting times."
    prompt = f"""
    You are an expert at verifying meeting times to make sure that they are in a valid and complete format

    Proposed time: {proposed_time}

    Question: {question}

    Example 1:
    Proposed time: Here is the proposed time: Monday, 10:00-10:30
    Valid: This output is valid

    Example 2:
    Proposed time: Monday, 10:00-10:30
    Invalid: This output is invalid because it's missing: Here is the proposed time:

    Is the proposed time valid, and in the correct format?
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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