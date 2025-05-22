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
    """Solve factual questions using a new approach: Concept Expansion and Evidence Ranking."""

    # Hypothesis: Expanding key concepts within the question and then ranking evidence based on its relevance to the *expanded* concepts will improve information retrieval and accuracy. This is fundamentally different from previous query-focused approaches.

    # Step 1: Concept Expansion (with examples)
    concept_expansion_prompt = f"""
    Expand the key concepts in the question with related terms and synonyms. Focus on adding concepts that would help a search engine find relevant results.
    Example 1:
    Question: What is the capital of Australia?
    Expanded Concepts: Australia: Canberra, cities in Australia, government of Australia, Australian territories; capital: administrative center, seat of government, federal capital

    Example 2:
    Question: In what year was Jamini Roy awarded the Padma Bhushan?
    Expanded Concepts: Jamini Roy: Indian painter, Bengali artist, Padma Bhushan: award, Indian honors, Indian civilian awards; year: date, time, when

    Question: {question}
    Expanded Concepts:
    """

    expanded_concepts = call_llm(concept_expansion_prompt, system_instruction="You are an expert concept expander.").strip()
    print(f"Expanded Concepts: {expanded_concepts}")

    # Step 2: Simulate Information Retrieval based on expanded concepts (with example)
    simulated_retrieval_prompt = f"""
    Simulate retrieving information based on the expanded concepts. Provide several sentences that would likely appear in search results.
    Example:
    Expanded Concepts: Australia: Canberra, cities in Australia; capital: administrative center
    Simulated Results: Canberra is the capital of Australia. Canberra is the seat of the Australian government. Sydney is a major city in Australia.

    Expanded Concepts: {expanded_concepts}
    Simulated Results:
    """
    simulated_results = call_llm(simulated_retrieval_prompt, system_instruction="You are a search engine simulating information retrieval.").strip()
    print(f"Simulated Results: {simulated_results}")

    # Step 3: Evidence Ranking (with example)
    evidence_ranking_prompt = f"""
    Rank the sentences based on their relevance to answering the original question, considering the expanded concepts. Assign a relevance score (1-10). Only include relevant sentences in your answer, if no sentences are relevant, respond with "No Relevant Sentences".
    Example:
    Question: What is the capital of Australia?
    Simulated Results: Canberra is the capital of Australia. Sydney is a major city in Australia.
    Ranked Evidence:
    1. Canberra is the capital of Australia (Relevance: 10)

    Question: {question}
    Simulated Results: {simulated_results}
    Ranked Evidence:
    """
    ranked_evidence = call_llm(evidence_ranking_prompt, system_instruction="You are an expert evidence ranker.").strip()
    print(f"Ranked Evidence: {ranked_evidence}")
    
    # Step 4: Extract Answer from Ranked Evidence (with Example)
    answer_extraction_prompt = f"""
    Extract the concise answer to the original question from the ranked evidence.
    Example:
    Question: What is the capital of Australia?
    Ranked Evidence:
    1. Canberra is the capital of Australia (Relevance: 10)
    Answer: Canberra

    Question: {question}
    Ranked Evidence: {ranked_evidence}
    Answer:
    """
    extracted_answer = call_llm(answer_extraction_prompt, system_instruction="You are a concise answer extractor.").strip()

    # Step 5: Validation (with Example)
    validation_prompt = f"""
    Validate that the extracted answer correctly and completely answers the original question.
    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Validation: VALID - Canberra is indeed the capital of Australia.

    Question: {question}
    Answer: {extracted_answer}
    Validation:
    """
    validation_result = call_llm(validation_prompt, system_instruction="You are a strict validator.").strip()

    if "VALID" in validation_result:
        return extracted_answer
    else:
        return "Could not be validated."