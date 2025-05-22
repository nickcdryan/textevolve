import os
import re
import math

# New Approach: Iterative Question Expansion and Answer Distillation with Dual Verification
# Hypothesis: By iteratively expanding the question with related context and distilling potential answers using a dual verification process (internal consistency and external plausibility), we can improve accuracy in factual question answering.
# This approach will test whether iteratively layering context helps refine the answer and if dual verification reduces errors.

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

def main(question, max_iterations=3):
    """Solve factual questions using iterative question expansion and answer distillation with dual verification."""

    expanded_question = question
    potential_answer = "No answer"

    for i in range(max_iterations):
        # Step 1: Iterative Question Expansion
        expansion_prompt = f"""
        Expand the question with related context and information.

        Example:
        Question: What is the capital of Australia?
        Expanded Question: What is the capital of the country Australia, and what is its significance?

        Question: {expanded_question}
        Expanded Question:
        """
        expanded_question = call_llm(expansion_prompt, system_instruction="You are an expert at expanding questions.").strip()

        # Step 2: Answer Extraction
        extraction_prompt = f"""
        Extract a concise answer from the expanded question.

        Example:
        Question: What is the capital of the country Australia, and what is its significance?
        Answer: Canberra

        Question: {expanded_question}
        Answer:
        """
        potential_answer = call_llm(extraction_prompt, system_instruction="You are an expert at concise answer extraction.").strip()

        # Step 3: Internal Consistency Verification
        internal_verification_prompt = f"""
        Verify if the answer is consistent with the information present in the expanded question.

        Example:
        Question: What is the capital of the country Australia, and what is its significance?
        Answer: Canberra
        Verification: Consistent - Canberra is mentioned as the capital in the question.

        Question: {expanded_question}
        Answer: {potential_answer}
        Verification:
        """
        internal_verification = call_llm(internal_verification_prompt, system_instruction="You are an expert at verifying internal consistency.").strip()

        if "Inconsistent" in internal_verification:
            potential_answer = "No answer"  # Reset if inconsistent

    # Step 4: External Plausibility Verification
    external_verification_prompt = f"""
    Verify if the potential answer is a plausible and accurate answer to the original question.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Verification: Plausible - Canberra is widely known as the capital of Australia.

    Question: {question}
    Answer: {potential_answer}
    Verification:
    """
    external_verification = call_llm(external_verification_prompt, system_instruction="You are an expert at verifying plausibility.").strip()

    if "Plausible" in external_verification:
        return potential_answer
    else:
        return "Could not be validated."