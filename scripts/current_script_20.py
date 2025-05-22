import os
import re
import math

# New Approach: Question Transformation and Knowledge Base Retrieval with Verification Loop
# Hypothesis: Transforming the question into a more specific query and using a verification loop after knowledge base retrieval will improve accuracy.
# This approach tests whether more specific questions and verification steps improves performance.

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

def main(question, max_attempts=3):
    """Solve factual questions using Question Transformation and Knowledge Base Retrieval with Verification Loop."""

    # Step 1: Question Transformation (with examples)
    transformation_prompt = f"""
    Transform the question into a more specific and targeted query for a knowledge base.

    Example 1:
    Original Question: What is the capital of Australia?
    Transformed Query: Return the capital city of the country Australia.

    Example 2:
    Original Question: Who choreographed Issey Miyake's produced “Aomori University Men’s Rhythmic Gymnastics Team” performance?
    Transformed Query: Give the name of the choreographer for the "Aomori University Men’s Rhythmic Gymnastics Team" performance produced by Issey Miyake.

    Original Question: {question}
    Transformed Query:
    """
    transformed_query = call_llm(transformation_prompt, system_instruction="You are an expert at question transformation.").strip()

    # Step 2: Knowledge Base Retrieval (simulated)
    knowledge_base_prompt = f"""
    Simulate retrieving information from a knowledge base based on the transformed query.

    Example:
    Query: Return the capital city of the country Australia.
    Knowledge Base Results: Canberra is the capital city of Australia.

    Query: {transformed_query}
    Knowledge Base Results:
    """
    knowledge_base_results = call_llm(knowledge_base_prompt, system_instruction="You are a helpful knowledge base.").strip()

    # Step 3: Verification Loop (with examples)
    answer = "Could not be validated." # Initialize
    for attempt in range(max_attempts):
        verification_prompt = f"""
        Verify if the knowledge base results provide a direct and accurate answer to the original question. If the extracted data contains the information sought in the original question, respond with the correct answer from the knowlege base results. If there isn't enough information, respond with 'insufficient information'.

        Example 1:
        Original Question: What is the capital of Australia?
        Knowledge Base Results: Canberra is the capital city of Australia.
        Answer: Canberra

        Example 2:
        Original Question: What is the wingspan of Eugnosta misella in millimeters?
        Knowledge Base Results: The wingspan of Eugnosta misella is 9-11 mm.
        Answer: 9-11 mm

        Original Question: {question}
        Knowledge Base Results: {knowledge_base_results}
        Answer:
        """
        answer = call_llm(verification_prompt, system_instruction="You are an expert validator.").strip()
        if answer != "insufficient information":
            break

    return answer