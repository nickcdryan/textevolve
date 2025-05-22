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

def extract_information(question):
    """Extract key information from the question, including entities and constraints."""
    system_instruction = "You are an expert information extractor."
    prompt = f"""
    Extract the key entities and constraints from the following question.

    Example 1:
    Question: What is the capital of the country where the Great Barrier Reef is located?
    Entities: Great Barrier Reef
    Constraints: Location is a country, seeking its capital

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Entities: Barcelona, Champions League, Milan, April 27, 2006
    Constraints: Corners taken by Barcelona, in that specific match

    Question: {question}
    Entities and Constraints:
    """
    return call_llm(prompt, system_instruction)

def generate_search_query(question, extracted_info):
    """Generate a search query based on the question and extracted information."""
    system_instruction = "You are a search query generator."
    prompt = f"""
    Generate a search query to answer the question, using the extracted information.

    Example 1:
    Question: What is the capital of Australia?
    Extracted Info: Australia, capital
    Search Query: "capital of Australia"

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Extracted Info: Barcelona, Champions League, Milan, April 27, 2006, corners
    Search Query: "Barcelona Milan Champions League April 27 2006 corner kicks"

    Question: {question}
    Extracted Info: {extracted_info}
    Search Query:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(question, search_results):
    """Extract the answer from the search results and provide a confidence score."""
    system_instruction = "You are an answer extraction expert."
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

    Question: {question}
    Search Results: {search_results}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, answer):
    """Validate if the extracted answer is correct and satisfies the question's requirements."""
    system_instruction = "You are an answer validator."
    prompt = f"""
    Validate if the extracted answer is correct and satisfies the question's requirements. Provide a detailed explanation.

    Example 1:
    Question: What is the capital of Australia?
    Answer: Canberra (Confidence: 10)
    Validation: VALID - The answer is correct and satisfies the question's requirements.

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Answer: 3 (Confidence: 10)
    Validation: VALID - The answer is correct and satisfies the question's requirements.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to answer the question."""
    try:
        # Step 1: Extract information
        extracted_info = extract_information(question)
        print(f"Extracted Info: {extracted_info}")

        # Step 2: Generate search query
        search_query = generate_search_query(question, extracted_info)
        print(f"Search Query: {search_query}")

        # Step 3: Simulate information retrieval
        search_results = call_llm(search_query, "You are a helpful search engine that provides concise, factual information.")
        print(f"Search Results: {search_results}")

        # Step 4: Extract answer
        extracted_answer_raw = extract_answer(question, search_results)
        print(f"Extracted Answer (raw): {extracted_answer_raw}")
        
        #Split out answer and confidence score
        try:
            extracted_answer = extracted_answer_raw.split('(Confidence:')[0].strip()
            confidence = int(extracted_answer_raw.split('(Confidence:')[1].replace(')','').strip())
        except:
            extracted_answer = extracted_answer_raw
            confidence = 5 #low confidence score to force validation to work

        # Step 5: Validate answer
        validation_result = validate_answer(question, extracted_answer)
        print(f"Validation Result: {validation_result}")

        if "VALID" in validation_result:
            return extracted_answer
        else:
            return "Could not be validated."
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"