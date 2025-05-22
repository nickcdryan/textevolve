import os
import re
import math

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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

def extract_information(question):
    """Extract key information from the question."""
    system_instruction = "You are an expert information extractor focusing on entities, constraints, and temporal context."
    prompt = f"""
    Extract the key entities, constraints, and temporal context from the following question.

    Example 1:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Entities: Great Barrier Reef
    Constraints: Location is a country, seeking its capital
    Temporal Context: None

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Entities: Barcelona, Champions League, Milan
    Constraints: Corners taken by Barcelona, in that specific match
    Temporal Context: April 27, 2006

    Example 3:
    Question: Who won the Eddington Medal in 1993?
    Entities: Eddington Medal
    Constraints: Seeking the winner
    Temporal Context: 1993

    Question: {question}
    Entities, Constraints, and Temporal Context:
    """
    return call_llm(prompt, system_instruction)

def generate_search_query(question, extracted_info):
    """Generate a search query."""
    system_instruction = "You are a search query generator, focusing on precision and temporal relevance."
    prompt = f"""
    Generate a search query to answer the question, using the extracted information.

    Example 1:
    Question: What is the capital of Australia?
    Extracted Info: Australia, capital, None
    Search Query: "capital of Australia"

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Extracted Info: Barcelona, Champions League, Milan, corners, April 27, 2006
    Search Query: "Barcelona Milan Champions League April 27 2006 corner kicks"

    Example 3:
    Question: Who won the Eddington Medal in 1993?
    Extracted Info: Eddington Medal, winner, 1993
    Search Query: "Eddington Medal winner 1993"

    Question: {question}
    Extracted Info: {extracted_info}
    Search Query:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, search_results):
    """Extract the answer with a confidence score."""
    system_instruction = "You are an answer extraction expert, focusing on precision and factual correctness."
    prompt = f"""
    Extract the answer to the question from the search results and provide a confidence score (1-10).

    Example 1:
    Question: What is the capital of Australia?
    Search Results: Canberra is the capital city of Australia.
    Answer: Canberra (Confidence: 10)

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Search Results: Barcelona took 3 corners in the match.
    Answer: 3 (Confidence: 10)

    Example 3:
    Question: Who won the Eddington Medal in 1993?
    Search Results: Leon Mestel won the Eddington Medal in 1993.
    Answer: Leon Mestel (Confidence: 10)

    Question: {question}
    Search Results: {search_results}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, answer):
    """Validate if the extracted answer is correct."""
    system_instruction = "You are a strict answer validator, focusing on factual correctness and temporal accuracy."
    prompt = f"""
    Validate if the extracted answer is correct and satisfies the question's requirements, including temporal context. Provide a detailed explanation.

    Example 1:
    Question: What is the capital of Australia?
    Answer: Canberra (Confidence: 10)
    Validation: VALID - The answer is correct. Australia's capital is Canberra.

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Answer: 3 (Confidence: 10)
    Validation: VALID - The answer is correct. Barcelona took 3 corners.

    Example 3:
    Question: Who won the Eddington Medal in 1993?
    Answer: Leon Mestel (Confidence: 10)
    Validation: VALID - The answer is correct. Leon Mestel won the Eddington Medal in 1993.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function."""
    try:
        # Extract information
        extracted_info = extract_information(question)
        print(f"Extracted Info: {extracted_info}")

        # Generate search query
        search_query = generate_search_query(question, extracted_info)
        print(f"Search Query: {search_query}")

        # Simulate information retrieval
        search_results = call_llm(search_query, "You are a helpful search engine that provides concise, factual information.")
        print(f"Search Results: {search_results}")

        # Extract answer
        extracted_answer_raw = extract_answer(question, search_results)
        print(f"Extracted Answer (raw): {extracted_answer_raw}")
        
        # Split out answer and confidence score
        try:
            extracted_answer = extracted_answer_raw.split('(Confidence:')[0].strip()
            confidence = int(extracted_answer_raw.split('(Confidence:')[1].replace(')','').strip())
        except:
            extracted_answer = extracted_answer_raw
            confidence = 5

        # Validate answer
        validation_result = validate_answer(question, extracted_answer)
        print(f"Validation Result: {validation_result}")

        if "VALID" in validation_result:
            return extracted_answer
        else:
            return "Could not be validated."
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"