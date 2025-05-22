import os
import re
import math

# Hypothesis: Implement a multi-agent system with a "Knowledge Navigator" that uses iterative refinement and source validation to address the problem of factual accuracy.
# The Knowledge Navigator will:
# 1. Formulate initial search queries
# 2. Evaluate initial search results for relevance and source credibility
# 3. Refine search queries based on initial results and source credibility
# 4. Extract a candidate answer and source
# 5. Validate the candidate answer against external knowledge and internal consistency
# 6. Use a second "Fact Checker" agent to confirm or deny findings.

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

def knowledge_navigator(question, max_iterations=3):
    """Navigate knowledge sources to find a reliable answer."""
    system_instruction = "You are a Knowledge Navigator, tasked with finding accurate information from multiple sources. You will analyze, refine, and validate information."

    search_query = question  # Initial search query
    candidate_answer = "No answer found." # initialize the candidate answer
    candidate_source = "None" # initialize the candidate source

    for i in range(max_iterations):
        # Step 1: Simulate search and extract potential answers

        search_results = call_llm(f"Search for: {search_query}", system_instruction="You are a search engine simulator. Provide concise, fact-based answers with source URLs.")

        # Step 2: Evaluate relevance and source credibility
        evaluation_prompt = f"""
        Evaluate these search results for relevance and source credibility.
        Question: {question}
        Search Results: {search_results}

        Example 1:
        Question: What is the capital of Australia?
        Search Results: Canberra is the capital of Australia. Source: wikipedia.org
        Relevance: Very Relevant
        Credibility: High

        Example 2:
        Question: What is the capital of Australia?
        Search Results: A blog post about visiting Sydney, Australia. Source: travelblog.com
        Relevance: Not Relevant
        Credibility: Low

        Relevance:
        Credibility:
        """

        evaluation = call_llm(evaluation_prompt, system_instruction="You are an expert at judging the relevancy and credibility of sources.")

        # Step 3: Extract potential answer

        extract_prompt = f"""
        From these search results, extract a concise answer and source.
        Question: {question}
        Search Results: {search_results}

        Example 1:
        Question: What is the capital of Australia?
        Search Results: Canberra is the capital of Australia. Source: wikipedia.org
        Answer: Canberra, Source: wikipedia.org

        Example 2:
        Question: Babymetal's song "Road of Resistance" charted at what number...?
        Search Results: Road of Resistance peaked at number 22 on the Billboard... Source: billboard.com
        Answer: 22, Source: billboard.com

        Answer:
        """
        extracted_answer = call_llm(extract_prompt, system_instruction="You are an expert answer extractor, focus on accuracy and succinctness.")

        if "Source:" in extracted_answer:
            candidate_answer = extracted_answer.split("Source:")[0].strip()
            candidate_source = extracted_answer.split("Source:")[1].strip()

        # Step 4: Refine search query (only if needed)
        if "Not Relevant" in evaluation:
            search_query = call_llm(f"Refine the search query for: {question}", system_instruction="You are an expert query refiner, use all known info to make queries more specific.")

    return candidate_answer, candidate_source

def fact_checker(question, answer, source):
    """Verify the answer with an external source."""
    system_instruction = "You are a Fact Checker, verifying information against reliable external sources."
    validation_prompt = f"""
    Verify this answer against a reliable external source.
    Question: {question}
    Answer: {answer}
    Source: {source}

    Example 1:
    Question: What is the capital of Australia?
    Answer: Canberra
    Source: wikipedia.org
    Validation: VALID

    Example 2:
    Question: Babymetal's song "Road of Resistance" charted at what number...?
    Answer: 22
    Source: billboard.com
    Validation: VALID

    Validation:
    """

    validation_result = call_llm(validation_prompt, system_instruction)
    return validation_result

def main(question):
    """Solve questions using a Knowledge Navigator and Fact Checker."""
    try:
        # Step 1: Knowledge Navigation

        candidate_answer, candidate_source = knowledge_navigator(question)
        print(f"Candidate answer is {candidate_answer}") # debug
        print(f"Candidate source is {candidate_source}") #debug
        # Step 2: Fact Checking

        validation_result = fact_checker(question, candidate_answer, candidate_source)
        print(f"Validation result is {validation_result}") #debug
        if "VALID" in validation_result:
            return candidate_answer
        else:
            return "Could not be validated."

    except Exception as e:
        return f"Error: {str(e)}"