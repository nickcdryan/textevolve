import os
import re
import math

def main(question):
    """
    This script implements a "Question Clarification & Focused Extraction" approach.
    The hypothesis is that by first clarifying the question's intent with the LLM,
    we can significantly improve the precision and relevance of information extraction,
    avoiding previous issues with misinterpretation and unnecessary decomposition.
    This will be followed by a verification step to ensure the extraction worked.
    """
    try:
        # Step 1: Clarify the question to better understand the user's intent
        clarification_result = clarify_question(question)
        if not clarification_result.get("is_valid"):
            return f"Error in question clarification: {clarification_result.get('validation_feedback')}"
        clarified_question = clarification_result["clarified_question"]
        print(f"Clarified question: {clarified_question}")

        # Step 2: Extract relevant information based on the clarified question
        extraction_result = extract_information(clarified_question, question)
        if not extraction_result.get("is_valid"):
            return f"Error in information extraction: {extraction_result.get('validation_feedback')}"
        extracted_info = extraction_result["extracted_info"]
        print(f"Extracted info: {extracted_info}")

        # Step 3: Synthesize the answer from extracted information
        answer = synthesize_answer(clarified_question, extracted_info)
        return answer

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def clarify_question(question, max_attempts=3):
    """Clarifies the question to better understand the user's intent."""
    system_instruction = "You are an expert at clarifying ambiguous questions."

    for attempt in range(max_attempts):
        clarification_prompt = f"""
        Given a question, rephrase it to be more specific and unambiguous, clarifying the user's intent.

        Example:
        Question: How many yards did the players combine for?
        Clarified Question: What was the total number of yards gained by all players mentioned in the passage?

        Question: {question}
        Clarified Question:
        """

        clarified_question = call_llm(clarification_prompt, system_instruction)

        verification_prompt = f"""
        Verify that the clarified question maintains the original intent while being more specific.

        Original Question: {question}
        Clarified Question: {clarified_question}

        Example:
        Original Question: How many yards did the players combine for?
        Clarified Question: What was the total number of yards gained by all players mentioned in the passage?
        Verification: Valid

        Is the clarified question valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "clarified_question": clarified_question}
        else:
            print(f"Clarification validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to clarify the question successfully."}

def extract_information(clarified_question, original_question, max_attempts=3):
    """Extracts relevant information from the passage based on the clarified question."""
    system_instruction = "You are an information extraction expert. Extract only the relevant information and nothing else."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract the information needed to answer the following question, using the original question for context.

        Original Question: {original_question}
        Question: {clarified_question}
        Extracted Information:
        """

        extracted_info = call_llm(extraction_prompt, system_instruction)

        verification_prompt = f"""
        Verify if the extracted information is relevant and complete for answering the question.

        Question: {clarified_question}
        Extracted Information: {extracted_info}

        Example:
        Question: What was the total number of yards gained by all players mentioned in the passage?
        Extracted Information: Player A gained 100 yards. Player B gained 50 yards.
        Verification: Valid

        Is the extracted information valid? Respond with 'Valid' or 'Invalid'.
        """

        verification_result = call_llm(verification_prompt, system_instruction)

        if "valid" in verification_result.lower():
            return {"is_valid": True, "extracted_info": extracted_info}
        else:
            print(f"Extraction validation failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    return {"is_valid": False, "validation_feedback": "Failed to extract the information successfully."}

def synthesize_answer(question, extracted_info):
    """Synthesizes the answer from the extracted information."""
    system_instruction = "You are an expert at synthesizing answers from extracted information. Provide a direct answer based on the extracted information."
    synthesis_prompt = f"""
        Given the question and the extracted information, create a final answer.

        Example:
        Question: What was the total number of yards gained by all players mentioned in the passage?
        Extracted Information: Player A gained 100 yards. Player B gained 50 yards.
        Answer: 150 yards

        Question: {question}
        Extracted Information: {extracted_info}
        Answer:
        """
    answer = call_llm(synthesis_prompt, system_instruction)
    return answer
def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os  # Import the os module

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