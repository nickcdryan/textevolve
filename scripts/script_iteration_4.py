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
    This script uses a fact-verification with self-correction approach.
    HYPOTHESIS: Implement a robust fact-verification stage using external knowledge and leverage self-correction capabilities to improve accuracy.
    This approach uses a main LLM call for generating an initial answer, then uses a separate LLM call to verify facts in that answer against the passage and a broader knowledge set.
    """

    # Step 1: Initial Answer Generation
    initial_prompt = f"""
    Provide a concise answer to the question based on the provided text.

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

    # Step 2: Fact Verification and Self-Correction
    verification_prompt = f"""
    Analyze the answer for factual correctness against the original question and a broader knowledge base.
    Identify any inaccuracies or inconsistencies. If the answer is incorrect, provide a corrected answer using the available information.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Proposed Answer: Tom Brady
    Analysis: Tom Brady is a quarterback, not a kicker. A more likely answer based on the context is a kicker like Josh Scobee. The passage will also be searched to verify this information.
    Corrected Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Proposed Answer: Gliese 915
    Analysis: The answer is correct. Gliese 915 is a white dwarf and known to have less mass than Nu Phoenicis based on general astronomical knowledge.
    Corrected Answer: Gliese 915

    Question: {question}
    Proposed Answer: {initial_answer}
    Analysis:
    """
    try:
        verification_response = call_llm(verification_prompt, "You are a fact-checker and self-correction expert.")
        # Attempt to extract the corrected answer; if impossible, stick with the initial answer
        if "Corrected Answer:" in verification_response:
          corrected_answer = verification_response.split("Corrected Answer:")[-1].strip()
        else:
          corrected_answer = initial_answer
    except Exception as e:
        print(f"Error during fact verification: {e}")
        corrected_answer = initial_answer # Fallback in case verification fails

    # Step 3: Final Output Validation (Ensure it's concise)
    final_validation_prompt = f"""
    Validate if the final answer is concise and accurately answers the question. Return the answer or a more concise version.

    Example 1:
    Question: Which player kicked the only field goal of the game?
    Proposed Answer: Josh Scobee was the player
    Final Answer: Josh Scobee

    Example 2:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Proposed Answer: Gliese 915, the white dwarf star.
    Final Answer: Gliese 915

    Question: {question}
    Proposed Answer: {corrected_answer}
    Final Answer:
    """

    try:
        final_answer = call_llm(final_validation_prompt, "You are a validator for concise answers.")
        final_answer = final_answer.strip()
        return final_answer
    except Exception as e:
        print(f"Error validating final answer: {e}")
        return corrected_answer # As a final safety