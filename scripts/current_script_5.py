import os
import re
import math

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

def main(question, max_attempts=3):
    """Solve factual questions using an iterative question refinement and information extraction approach."""

    # Hypothesis: Refining the question itself based on initial retrieval failures, and using this refined question to get more specific information, will improve accuracy.
    # This addresses the passive behavior and insufficient context detection issues from previous iterations.

    # Step 1: Generate an initial search query based on the question (with examples)
    initial_search_query_prompt = f"""
    Given a factual question, generate a concise and effective initial search query.

    Example 1:
    Question: What was the first name of the wife of the American chemist Ralph E. Oesper?
    Search Query: Ralph E. Oesper wife

    Example 2:
    Question: Who formed the Dubai-based band Sho? in June 2009?
    Search Query: Dubai band Sho formed 2009
    
    Question: {question}
    Search Query:
    """
    initial_search_query = call_llm(initial_search_query_prompt, "You are a search query generator.")

    # Step 2: Simulate information retrieval with a limited context (with an example)
    retrieved_info = f"Simulated web search results for: {initial_search_query}. Limited context available."  # Replace with actual search API call
    
    # Step 3: Determine if the retrieved info is sufficient to answer the question (with example and validation)
    sufficiency_check_prompt = f"""
    Given a question and retrieved information, determine if the information is sufficient to answer the question.

    Example:
    Question: What was the first name of the wife of the American chemist Ralph E. Oesper?
    Retrieved Information: Ralph E. Oesper's wife was a chemist.
    Sufficient: No. The first name is missing.

    Question: {question}
    Retrieved Information: {retrieved_info}
    Sufficient:
    """
    sufficiency_result = call_llm(sufficiency_check_prompt, "You are a helpful expert at assessing information sufficiency.")

    # Step 4: If not sufficient, refine the question (with examples)
    if "No" in sufficiency_result:
        refine_question_prompt = f"""
        Given a question and the reason why the initial information was insufficient, refine the question to get a more specific answer.
        
        Example:
        Original Question: What was the first name of the wife of the American chemist Ralph E. Oesper?
        Reason: The first name is missing.
        Refined Question: What was the *first name* of Ralph E. Oesper's wife?

        Question: {question}
        Reason: {sufficiency_result}
        Refined Question:
        """
        refined_question = call_llm(refine_question_prompt, "You are an expert at refining questions.")

        # Step 5: Retrieve information using refined question.
        refined_search_query_prompt = f"""
        Given a refined question, generate a search query.
        Question: {refined_question}
        Search Query:
        """
        refined_search_query = call_llm(refined_search_query_prompt, "You are an search query generator.")

        retrieved_info = f"Simulated web search results for: {refined_search_query}. Specific information available."
    else:
        refined_question = question # If the sufficiency test passed

    # Step 6: Extract the answer from retrieved information (with examples)
    answer_extraction_prompt = f"""
    Given a question and retrieved information, extract the answer.
    Example:
    Question: What was the *first name* of Ralph E. Oesper's wife?
    Relevant Information: Helen Oesper was the wife of Ralph E. Oesper.
    Answer: Helen

    Question: {refined_question}
    Relevant Information: {retrieved_info}
    Answer:
    """
    extracted_answer = call_llm(answer_extraction_prompt, "You are an expert question answering system.")
    
    # Step 7: Verify answer with original question
    verification_prompt = f"""
    Verify that the following answer accurately addresses the *original* question:
    Original question: {question}
    Extracted Answer: {extracted_answer}
    Verification (Correct/Incorrect):
    """
    verification_result = call_llm(verification_prompt, "You are a validation expert")

    if "Correct" in verification_result:
        return extracted_answer
    else:
        return "Could not be validated."