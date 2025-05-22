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

def generate_query_and_validate(question, max_attempts=3):
    """
    Generates a search query from a question and validates its effectiveness by assessing
    if the top search snippets contain key entities and relationships needed to answer the question.
    Returns both the generated query and top search snippets.
    """
    system_instruction_query_gen = "You are an expert at generating effective search queries that help answer questions."
    system_instruction_search_validator = "You are an expert at validating whether a set of search snippets are relevant to answering the question"
    # Hypothesis: By generating and validating the query BEFORE retrieving the information, we can significantly improve the information retrieval and hallucination problems that are causing the pipeline to fail
    for attempt in range(max_attempts):
        # Step 1: Generate Search Query with Examples
        query_prompt = f"""
        Generate a search query to retrieve information needed to answer the question.

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Query: Ralph E. Oesper first name

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Query: Maharaj Kishan Bhan Padma Bhushan year

        Question: {question}
        Search Query:
        """
        search_query = call_llm(query_prompt, system_instruction_query_gen)
        # Step 2: Simulate Retrieving Top Search Snippets - IMPORTANT: IN A REAL SYSTEM THIS WOULD BE SEARCH API
        search_snippets = call_llm(f"Provide top 3 search snippets for: {search_query}", "You are a helpful search engine providing realistic search results.")

        # Step 3: Validate Relevance of Search Snippets with Examples
        validation_prompt = f"""
        Determine if the following search snippets are relevant to answering the question. If they are, respond with "RELEVANT: [brief explanation]". If not, respond with "IRRELEVANT: [detailed explanation]".

        Example 1:
        Question: What was the first name of Ralph E. Oesper?
        Search Snippets: Ralph Oesper was a professor...; His middle name was E...; There is no information on his first name.
        Validation: IRRELEVANT: The snippets don't reveal his first name.

        Example 2:
        Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
        Search Snippets: Maharaj Kishan Bhan received the Padma Bhushan in 2013; He was a scientist; He worked in civil services.
        Validation: RELEVANT: Snippets contain MKB and the year he received the award

        Question: {question}
        Search Snippets: {search_snippets}
        Validation:
        """
        validation_result = call_llm(validation_prompt, system_instruction_search_validator)

        if "RELEVANT:" in validation_result:
            return search_query, search_snippets # Return both the search query and relevant context
        else:
            print(f"Attempt {attempt + 1}: Search snippets deemed irrelevant. Trying again...")

    return None, None  # Return None if no relevant context is found
def generate_answer_with_snippets(question, search_snippets):
    """
    Generates an answer using the validated search snippets, ensuring that the answer
    is directly supported by the information in the snippets.
    """
    system_instruction = "You are an expert at answering question given relevant search snippets"
    # Now we leverage the search snippets to answer the question directly
    answer_prompt = f"""
    Answer the question using ONLY the information present in the search snippets.

    Example 1:
    Question: What was the first name of Ralph E. Oesper?
    Search Snippets: No results found.
    Answer: Answer not found.

    Example 2:
    Question: In which year did Maharaj Kishan Bhan receive the Padma Bhushan for civil services?
    Search Snippets: Maharaj Kishan Bhan was awarded the Padma Bhushan in 2013.; He was a famous scientist.
    Answer: 2013

    Question: {question}
    Search Snippets: {search_snippets}
    Answer:
    """
    answer = call_llm(answer_prompt, system_instruction)
    return answer

def main(question):
    """
    Main function to orchestrate the validated query generation, information retrieval (simulated),
    and answer generation process.
    """
    search_query, search_snippets = generate_query_and_validate(question)

    if search_query and search_snippets:
        answer = generate_answer_with_snippets(question, search_snippets)
        return answer
    else:
        return "Answer not found." # If not able to retrieve reliable context then return not found