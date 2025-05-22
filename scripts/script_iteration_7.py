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
    """Solve factual questions using a new approach: Decomposition and targeted retrieval with active feedback."""

    # Hypothesis: Answering questions effectively requires a focused retrieval strategy, explicitly asking for *what we don't know* to guide the search process. This strategy will actively seek missing information, improving on the passive behavior observed in previous attempts.
    # 1. Decompose into known and unknown
    # 2. Target retrieval based on what is unknown
    # 3. Iteratively refine search strategy by focusing on what is *still* unknown

    # Step 1: Decompose question into KNOWN and UNKNOWN elements (with examples)
    decomposition_prompt = f"""
    Decompose the following question into what information is ALREADY KNOWN and what information is UNKNOWN and needs to be retrieved.

    Example 1:
    Question: What is the capital of Australia?
    Known: Australia, has a capital
    Unknown: The name of the capital

    Example 2:
    Question: In what year did Etta Cone last visit Europe?
    Known: Etta Cone visited Europe
    Unknown: The specific year of her last visit

    Question: {question}
    Known:
    Unknown:
    """

    decomposition_result = call_llm(decomposition_prompt, system_instruction="You are an expert at breaking down questions.")

    try:
        known = decomposition_result.split('Unknown:')[0].replace('Known:','').strip()
        unknown = decomposition_result.split('Unknown:')[1].strip()
    except:
        return "Error in decomposing knowns and unknowns"
    print (f"Known: {known}")
    print (f"Unknown: {unknown}")

    # Step 2: Generate targeted retrieval query focused on the UNKNOWN (with examples)
    targeted_query_prompt = f"""
    Based on the KNOWN and UNKNOWN elements, generate a highly targeted search query focused on retrieving the UNKNOWN.

    Example 1:
    Known: Australia, has a capital
    Unknown: The name of the capital
    Query: "capital of Australia"

    Example 2:
    Known: Etta Cone visited Europe
    Unknown: The specific year of her last visit
    Query: "Etta Cone last visit Europe year"

    Known: {known}
    Unknown: {unknown}
    Query:
    """

    targeted_query = call_llm(targeted_query_prompt, system_instruction="You are an expert at generating highly targeted search queries.")
    print (f"Targeted Query: {targeted_query}")

    # Step 3: Simulate Retrieval
    retrieved_info = f"Simulated web search results for: {targeted_query}. Placeholder for real search functionality."

    # Step 4: Extract the answer (with examples)
    answer_extraction_prompt = f"""
    Given the original question and the retrieved information, extract a CONCISE answer.
    Example 1:
    Question: What is the capital of Australia?
    Retrieved Info: Canberra is the capital city of Australia.
    Answer: Canberra

    Example 2:
    Question: In what year did Etta Cone last visit Europe?
    Retrieved Info: Etta Cone's last visit to Europe was in 1951.
    Answer: 1951

    Question: {question}
    Retrieved Info: {retrieved_info}
    Answer:
    """
    extracted_answer = call_llm(answer_extraction_prompt, system_instruction="You are an expert at concise answer extraction.")
    print (f"Extracted Answer: {extracted_answer}")

    # Step 5: VALIDATION: Does extracted answer actually answer the question?
    validation_prompt = f"""
    Does the extracted answer actually answer the ORIGINAL question? Respond with "YES" or "NO".

    Question: {question}
    Extracted Answer: {extracted_answer}

    Example 1:
    Question: What is the capital of Australia?
    Extracted Answer: Canberra
    Does the extracted answer actually answer the ORIGINAL question? Respond with "YES" or "NO".
    YES

    Example 2:
    Question: In what year did Etta Cone last visit Europe?
    Extracted Answer: 1951
    Does the extracted answer actually answer the ORIGINAL question? Respond with "YES" or "NO".
    YES

    Original Question: {question}
    Extracted Answer: {extracted_answer}
    Does the extracted answer actually answer the ORIGINAL question? Respond with "YES" or "NO".
    """
    validation_result = call_llm(validation_prompt, system_instruction="You are an expert validator who determines if the question is answered.").strip()

    if "YES" in validation_result.upper():
      return extracted_answer
    else:
      return "Could not be validated. Not an answer."