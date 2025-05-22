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
    """Solve factual questions using a fact verification with multi-source integration approach."""

    # Hypothesis: Explicitly searching for validating sources and integrating information from multiple sources before answering will increase accuracy. This addresses the previous issues of inaccurate knowledge retrieval and ineffective information extraction.

    # Step 1: Generate multiple search queries (n=3) to find validating sources.
    search_query_prompt = f"""
    Generate three diverse search queries to find independent validating sources for the following question.

    Example 1:
    Question: What is the name of the individual who was awarded the Paul Karrer Gold Medal in 2004?
    Queries:
    1. "Paul Karrer Gold Medal 2004 recipient"
    2. "Who won Paul Karrer Gold Medal 2004"
    3. "Awardees of Paul Karrer Gold Medal in 2004"

    Question: {question}
    Queries:
    """
    search_queries = call_llm(search_query_prompt, system_instruction="You are an expert at generating diverse search queries.").split("\n")

    # Step 2: Simulate retrieval of context from the web for each query and VERIFY that a source exists
    retrieved_contexts = []
    for query in search_queries:
        context = f"Simulated web search results for: {query}. Placeholder for real search functionality."
        # Verify that results are not empty
        verification_prompt = f"""Question: {question} Search query: {query}. Retrieved context: {context}. Is the context useful to answering the question? Answer 'yes' or 'no'."""
        verification_result = call_llm(verification_prompt, "Validating retrieved context")
        retrieved_contexts.append(context if "yes" in verification_result.lower() else "No relevant context found.")
    
    # Step 3: Extract answers from *each* context, and then synthesize them.
    answer_extraction_prompt = f"""
    Given the question and retrieved contexts from multiple sources, extract an answer from each. Then, synthesize a final answer, considering the consistency and reliability of the sources.
    Question: {question}

    Context 1: {retrieved_contexts[0]}
    Context 2: {retrieved_contexts[1]}
    Context 3: {retrieved_contexts[2]}

    Example 1:
    Question: What is the capital of Australia?
    Context 1: Canberra is the capital city of Australia.
    Context 2: Australia's capital is Canberra.
    Context 3: Canberra serves as the capital of the Commonwealth of Australia.
    Answer: Canberra, based on multiple consistent sources.

    Answer:
    """
    final_answer = call_llm(answer_extraction_prompt, system_instruction="You are an expert at extracting and synthesizing answers from multiple sources.")

    # Step 4: Final validation that the synthesized answer answers the question
    validation_prompt = f"""
    Validate that the following extracted and synthesized answer correctly answers the question.

    Question: {question}
    Answer: {final_answer}

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra, based on multiple consistent sources.
    Validation: Correct; Canberra is the capital of Australia.

    Validation:
    """

    validation_result = call_llm(validation_prompt, system_instruction="You are an expert answer validator.")

    if "Correct" in validation_result:
        return final_answer
    else:
        return "Could not be validated."