import os
import re
import math # for react
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

def retrieve_relevant_context(question, max_attempts=3):
    """
    Retrieve relevant context for the given question using LLM-based search query generation and a simulated search engine.
    This directly addresses the hallucination and inaccurate retrieval issues from previous iterations.
    Includes a validation loop to ensure the retrieved context is relevant.
    """
    system_instruction = "You are an expert at generating search queries to retrieve relevant information."

    for attempt in range(max_attempts):
        # Step 1: Generate search query with examples
        query_prompt = f"""
        Generate a concise search query to find relevant information for the given question.

        Example 1:
        Question: In which year was Jamini Roy (an Indian painter) awarded the Padma Bhushan by the Government of India?
        Search Query: "Jamini Roy Padma Bhushan award year"

        Example 2:
        Question: Which architect was tasked with finishing the chapel in the newly built Papal apartment when its construction remained incomplete after Pope Paul IV moved in, in October 1556?
        Search Query: "architect finishing Papal apartment Pope Paul IV 1556"

        Example 3:
        Question: In 1993, Vaughan Jones was elected to which academy?
        Search Query: "Vaughan Jones elected academy 1993"

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction)

        # Step 2: Simulate search and retrieve context
        context = call_llm(f"Provide concise information about: {search_query}", "You are a helpful search engine.")

        # Step 3: Validate context relevance with examples
        validation_prompt = f"""
        Validate if the retrieved context is relevant to the question.
        If relevant, respond with "RELEVANT: [brief explanation]".
        If not relevant, respond with "IRRELEVANT: [detailed explanation]".

        Example 1:
        Question: In which year was Jamini Roy awarded the Padma Bhushan?
        Context: Jamini Roy received the Padma Bhushan in 1954.
        Validation: RELEVANT: The context directly answers the question about the award year.

        Example 2:
        Question: Which architect finished the Papal apartment chapel in 1556?
        Context: Pirro Ligorio completed the chapel in the Papal apartment in October 1556.
        Validation: RELEVANT: The context identifies the architect and the task.

        Question: {question}
        Context: {context}
        Validation:
        """
        validation_result = call_llm(validation_prompt, "You are an expert at determining the relevance of a text to a question.")

        if "RELEVANT:" in validation_result:
            return context
        else:
            print(f"Attempt {attempt + 1}: Retrieved context is irrelevant. Retrying...")

    return "No relevant context found." # Fallback after multiple attempts

def generate_answer_with_context(question, context):
    """
    Generate the answer using the retrieved context.
    This leverages the LLM's reasoning capabilities to synthesize an answer based on the context.
    """
    system_instruction = "You are an expert at answering questions based on provided context."

    prompt = f"""
    Answer the question using the provided context. If the context does not contain the answer, state "Answer not found in context."

    Example 1:
    Question: In which year was Jamini Roy awarded the Padma Bhushan?
    Context: Jamini Roy received the Padma Bhushan in 1954.
    Answer: 1954

    Example 2:
    Question: Which architect finished the Papal apartment chapel in 1556?
    Context: Pirro Ligorio completed the chapel in the Papal apartment in October 1556.
    Answer: Pirro Ligorio

    Question: {question}
    Context: {context}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """
    Main function: Orchestrates context retrieval and answer generation.
    """
    context = retrieve_relevant_context(question)
    if "No relevant context found" in context:
        return "Could not find the answer."

    answer = generate_answer_with_context(question, context)
    return answer