import os
import re
import math

# Hypothesis: A "Chain of Thought with Expert Roles and Validation" approach to improve information extraction and validation.
# The key idea is to use specialized LLM roles for each step of the process (extraction, synthesis, validation)
# and embed validation steps after each to improve overall reliability.

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

def extract_info(question):
    """Extract key information from the question using a specialized LLM."""
    system_instruction = "You are an information extraction expert, skilled at identifying key details from questions."
    prompt = f"""
    Extract the key entities, constraints, and expected answer format from the question.

    Example:
    Question: In what year did Charlton Publications sell Hit Parader?
    Extracted Information:
    {{
        "entities": ["Charlton Publications", "Hit Parader"],
        "constraints": [],
        "answer_format": "year"
    }}

    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Extracted Information:
    {{
        "entities": ["Asus ROG Phone 5s Pro"],
        "constraints": ["peak brightness"],
        "answer_format": "numerical value with unit nits"
    }}

    Question: {question}
    Extracted Information:
    """
    return call_llm(prompt, system_instruction)

def validate_extraction(question, extracted_info):
    """Validate the extracted information."""
    system_instruction = "You are a strict validation expert who validates extracted information."
    prompt = f"""
    Validate if the extracted information correctly captures the key entities, constraints, and answer format.

    Example:
    Question: In what year did Charlton Publications sell Hit Parader?
    Extracted Information: {{ "entities": ["Charlton Publications", "Hit Parader"], "constraints": [], "answer_format": "year" }}
    Validation: VALID

    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Extracted Information: {{ "entities": ["Asus ROG Phone 5s Pro"], "constraints": ["peak brightness"], "answer_format": "numerical value with unit nits" }}
    Validation: VALID

    Question: {question}
    Extracted Information: {extracted_info}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def generate_search_query(extracted_info):
    """Generate a search query based on the extracted information."""
    system_instruction = "You are an expert search query generator, designing effective queries to find answers."
    prompt = f"""
    Generate a search query based on the extracted information, focusing on retrieving factual answers.

    Example:
    Extracted Information: {{ "entities": ["Charlton Publications", "Hit Parader"], "constraints": [], "answer_format": "year" }}
    Search Query: "Charlton Publications sell Hit Parader year"

    Extracted Information: {{ "entities": ["Asus ROG Phone 5s Pro"], "constraints": ["peak brightness"], "answer_format": "numerical value with unit nits" }}
    Search Query: "Asus ROG Phone 5s Pro peak brightness nits"

    Extracted Information: {extracted_info}
    Search Query:
    """
    return call_llm(prompt, system_instruction)

def retrieve_info(search_query):
    """Retrieve relevant information using the generated search query."""
    system_instruction = "You are a search engine simulator providing factual and concise information."
    prompt = f"""
    Simulate search results for the query.

    Example:
    Search Query: "Charlton Publications sell Hit Parader year"
    Search Results: Charlton Publications sold Hit Parader in 1991.

    Search Query: "Asus ROG Phone 5s Pro peak brightness nits"
    Search Results: The peak brightness of the Asus ROG Phone 5s Pro is 1200 nits.

    Search Query: {search_query}
    Search Results:
    """
    return call_llm(prompt, system_instruction)

def extract_answer(retrieved_info, extracted_info):
    """Extract the answer from the retrieved information, considering the expected format."""
    system_instruction = "You are an answer extraction expert, focusing on accuracy and formatting."
    prompt = f"""
    Extract the answer from the retrieved information, based on the expected answer format.

    Example:
    Retrieved Information: Charlton Publications sold Hit Parader in 1991.
    Extracted Information: {{ "entities": ["Charlton Publications", "Hit Parader"], "constraints": [], "answer_format": "year" }}
    Answer: 1991

    Retrieved Information: The peak brightness of the Asus ROG Phone 5s Pro is 1200 nits.
    Extracted Information: {{ "entities": ["Asus ROG Phone 5s Pro"], "constraints": ["peak brightness"], "answer_format": "numerical value with unit nits" }}
    Answer: 1200 nits

    Retrieved Information: {retrieved_info}
    Extracted Information: {extracted_info}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, answer):
    """Validate the extracted answer against the question."""
    system_instruction = "You are a fact validator, ensuring the answer is correct and complete."
    prompt = f"""
    Validate if the answer accurately and completely answers the question.

    Example:
    Question: In what year did Charlton Publications sell Hit Parader?
    Answer: 1991
    Validation: VALID

    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Answer: 1200 nits
    Validation: VALID

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Orchestrate the question-answering process."""
    try:
        # 1. Extract Information
        extracted_info = extract_info(question)
        validation_result = validate_extraction(question, extracted_info)

        if "VALID" not in validation_result:
            return "Could not extract information."

        # 2. Generate Search Query
        search_query = generate_search_query(extracted_info)

        # 3. Retrieve Information
        retrieved_info = retrieve_info(search_query)

        # 4. Extract Answer
        answer = extract_answer(retrieved_info, extracted_info)

        # 5. Validate Answer
        final_validation = validate_answer(question, answer)

        if "VALID" not in final_validation:
            return "Could not be validated."

        return answer

    except Exception as e:
        return f"Error: {str(e)}"