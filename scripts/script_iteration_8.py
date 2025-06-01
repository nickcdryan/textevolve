import os
import re
import math
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

def main(question):
    """
    EXPLORATION: This script implements a **Iterative Solution Refinement with Multi-faceted Validation**.

    HYPOTHESIS: Instead of relying on a single verification or debate loop, this system will iteratively
    refine the solution through multiple validation steps, each focusing on a different aspect
    of the problem (e.g., numerical accuracy, logical consistency, completeness). This multi-faceted
    validation will lead to a more robust and accurate final answer. This attempts to explicitly
    address numerical accuracy which has been identified as a key failure point in the past.

    This approach is DIFFERENT from previous iterations by:
    1. Introducing a dedicated numerical accuracy validation step.
    2. Using multi-faceted validations to address different aspects of the problem.
    3. A separate 'relevance checker' to verify that the solution is related and only contains content relating to the question asked, as the agents in the past have failed by hallucinating.
    """

    # Step 1: Initial Solution Generation
    initial_prompt = f"""
    Provide a concise and accurate answer to the following question, extracting information
    directly from the provided text. Focus on identifying the key entities and relationships.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Answer: Gliese 915

    Question: {question}
    Answer:
    """

    try:
        initial_answer = call_llm(initial_prompt, "You are a precise information retriever.")
        initial_answer = initial_answer.strip()
    except Exception as e:
        print(f"Error generating initial answer: {e}")
        return "Error generating initial answer."

    # Step 2: Numerical Accuracy Validation (if applicable)
    numerical_validation_prompt = f"""
    If the question involves numerical values or calculations, verify that the answer
    is numerically accurate based on the provided text. If there are calculations, show them.
    If the answer is not numerically accurate, provide a corrected answer.
    If the question is not about numbers, simply state "Not applicable."

    Example 1:
    Question: How many yards longer was the longest touchdown pass than the longest field goal?
    Proposed Answer: 32
    Verification: Longest touchdown pass: 80 yards, Longest field goal: 48 yards. 80 - 48 = 32. The answer is accurate.
    Corrected Answer: 32

    Example 2:
    Question: Which player kicked the only field goal of the game?
    Proposed Answer: Josh Scobee
    Verification: Not applicable.
    Corrected Answer: Josh Scobee

    Question: {question}
    Proposed Answer: {initial_answer}
    Verification:
    """

    try:
        numerical_validation = call_llm(numerical_validation_prompt, "You are a numerical accuracy expert.")
        if "Not applicable" not in numerical_validation:
            corrected_answer = numerical_validation.split("Corrected Answer:")[-1].strip()
        else:
            corrected_answer = initial_answer
    except Exception as e:
        print(f"Error during numerical validation: {e}")
        corrected_answer = initial_answer # Fallback

    # Step 3: Logical Consistency Validation
    logical_validation_prompt = f"""
    Verify that the answer is logically consistent with the question and the provided text.
    If the answer is logically inconsistent, provide a revised answer.

    Example:
    Question: Which player kicked the only field goal of the game?
    Proposed Answer: Tom Brady
    Revised Answer: Josh Scobee

    Question: {question}
    Proposed Answer: {corrected_answer}
    Revised Answer:
    """
    try:
        logical_validation_response = call_llm(logical_validation_prompt, "You are a logical consistency expert.")
        revised_answer = logical_validation_response.split("Revised Answer:")[-1].strip()
    except Exception as e:
        print(f"Error in logical consistency validation: {e}")
        revised_answer = corrected_answer

    # Step 4: Solution Relevance Validation
    relevance_validation_prompt = f"""
    Confirm the provided solution only responds to the question asked. Remove all hallucinated text.

    Question: What year was Barack Obama inaugurated?
    Solution: Barack Obama became President in 2009, a significant year in American history as the first black president took office. He passed the affordable care act.
    Confirmed Solution: 2009

    Question: {question}
    Solution: {revised_answer}
    Confirmed Solution:
    """
    try:
        relevance_response = call_llm(relevance_validation_prompt, "You are a relevance expert.")
        final_answer = relevance_response.split("Confirmed Solution:")[-1].strip()
    except Exception as e:
        print(f"Error in relevance validation: {e}")
        final_answer = revised_answer

    # Step 5: Return Final Answer
    return final_answer