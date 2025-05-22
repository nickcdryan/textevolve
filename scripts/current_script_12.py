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
    """Solve factual questions using a new approach: Knowledge Base Selection and Targeted Fact Verification."""

    # Hypothesis: Prioritizing the selection of the most relevant simulated knowledge base BEFORE query formulation and then employing targeted fact verification against that knowledge base will improve accuracy. This contrasts with previous approaches that focused on query refinement or concept expansion.
    #This approach seeks to minimize errors caused by using the wrong context from the start.

    # Step 1: Knowledge Base Selection (with examples)
    kb_selection_prompt = f"""
    Select the most relevant knowledge base for answering the question. Choose ONE of the following options.

    Options:
    1. General Knowledge
    2. Historical Events
    3. Scientific Data
    4. Biographical Information
    5. Geographical Data

    Example 1:
    Question: What is the capital of Australia?
    Knowledge Base: Geographical Data

    Example 2:
    Question: Who was the lead programmer for Project Firebreak?
    Knowledge Base: Biographical Information

    Question: {question}
    Knowledge Base:
    """
    selected_kb = call_llm(kb_selection_prompt, system_instruction="You are an expert at selecting the most relevant knowledge base.").strip()
    print(f"Selected Knowledge Base: {selected_kb}")

    # Step 2: Generate Targeted Search Query (with examples, based on selected KB)
    query_generation_prompt = f"""
    Generate a search query tailored to the selected knowledge base that will help answer the question.

    Knowledge Base: {selected_kb}
    Example 1:
    Question: What is the capital of Australia?
    Knowledge Base: Geographical Data
    Query: "capital of Australia geographical data"

    Example 2:
    Question: Who was the lead programmer for Project Firebreak?
    Knowledge Base: Biographical Information
    Query: "lead programmer Project Firebreak biographical information"

    Question: {question}
    Query:
    """
    targeted_query = call_llm(query_generation_prompt, system_instruction="You are an expert at generating targeted search queries.").strip()
    print(f"Targeted Query: {targeted_query}")

    # Step 3: Simulate Information Retrieval (Simulate retrieval from selected KB)
    retrieved_info = call_llm(f"Simulated web search results from {selected_kb} for: {targeted_query}. Focus on concise and relevant results.", "You are a search engine.")
    print(f"Retrieved Information: {retrieved_info}")

    # Step 4: Extract Answer from Retrieved Information (with examples)
    answer_extraction_prompt = f"""
    Extract the concise answer to the original question from the retrieved information.
    Example 1:
    Question: What is the capital of Australia?
    Retrieved Info: Canberra is the capital of Australia.
    Answer: Canberra

    Example 2:
    Question: Who was the lead programmer for Project Firebreak?
    Retrieved Info: John Smith was the lead programmer for Project Firebreak.
    Answer: John Smith

    Question: {question}
    Retrieved Info: {retrieved_info}
    Answer:
    """
    extracted_answer = call_llm(answer_extraction_prompt, system_instruction="You are an expert at concise answer extraction.").strip()
    print(f"Extracted Answer: {extracted_answer}")

    # Step 5: Targeted Fact Verification (Validate answer against selected KB)
    fact_verification_prompt = f"""
    Verify if the extracted answer is a factually correct and complete answer to the original question, given the knowledge base. Respond with "VALID" or "INVALID".
    Knowledge Base: {selected_kb}
    Example 1:
    Question: What is the capital of Australia?
    Answer: Canberra
    Verification: VALID

    Example 2:
    Question: Who was the lead programmer for Project Firebreak?
    Answer: John Smith
    Verification: VALID

    Question: {question}
    Answer: {extracted_answer}
    Verification:
    """
    verification_result = call_llm(fact_verification_prompt, system_instruction="You are a strict fact verifier.").strip()

    if "VALID" in verification_result:
        return extracted_answer
    else:
        return "Could not be validated."