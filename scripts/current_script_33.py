import os
import re
import math

# Hypothesis: Implement a "Knowledge Source Navigator with Chain-of-Verification" approach, focusing on targeted search query generation,
# multi-example prompting, and intermediate validation to enhance answer accuracy. The primary change is using multiple few-shot examples in the prompt, and adding validation checks through the different parts of the pipeline.

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

def generate_search_query(question):
    """Generate a targeted search query based on the question."""
    system_instruction = "You are an expert search query generator, designing effective queries to find answers."
    prompt = f"""
    Generate a targeted search query based on the question.
    
    Example 1:
    Question: In what year did Charlton Publications sell Hit Parader?
    Search Query: "Charlton Publications" sold "Hit Parader" year
    
    Example 2:
    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Search Query: "Asus ROG Phone 5s Pro" peak brightness nits
    
    Example 3:
    Question: Who played Creon in Antigone at the Epidaurus Festival 2022?
    Search Query: Creon Antigone Epidaurus Festival 2022 cast
    
    Question: {question}
    Search Query:
    """
    search_query = call_llm(prompt, system_instruction)
    print(f"Generated search query: {search_query}") # Debugging
    return search_query

def retrieve_info(search_query):
    """Retrieve relevant information using the generated search query."""
    system_instruction = "You are a search engine simulator providing factual and concise information."
    prompt = f"""
    Simulate search results for the query.
    
    Example 1:
    Search Query: "Charlton Publications" sold "Hit Parader" year
    Search Results: Charlton Publications sold Hit Parader in 1991.
    
    Example 2:
    Search Query: "Asus ROG Phone 5s Pro" peak brightness nits
    Search Results: The peak brightness of the Asus ROG Phone 5s Pro is 1200 nits.
    
    Example 3:
    Search Query: Creon Antigone Epidaurus Festival 2022 cast
    Search Results: Vasilis Bisbikis played Creon in Antigone at the Epidaurus Festival in 2022.
    
    Search Query: {search_query}
    Search Results:
    """
    retrieved_info = call_llm(prompt, system_instruction)
    print(f"Retrieved info: {retrieved_info}") # Debugging
    return retrieved_info

def extract_answer(question, retrieved_info):
    """Extract the answer from the retrieved information."""
    system_instruction = "You are an expert at extracting precise answers from text. Focus on accuracy."
    prompt = f"""
    Extract the concise answer from the search results.
    
    Example 1:
    Question: In what year did Charlton Publications sell Hit Parader?
    Search Results: Charlton Publications sold Hit Parader in 1991.
    Answer: 1991
    
    Example 2:
    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Search Results: The peak brightness of the Asus ROG Phone 5s Pro is 1200 nits.
    Answer: 1200 nits
    
    Example 3:
    Question: Who played Creon in Antigone at the Epidaurus Festival 2022?
    Search Results: Vasilis Bisbikis played Creon in Antigone at the Epidaurus Festival in 2022.
    Answer: Vasilis Bisbikis
    
    Question: {question}
    Search Results: {retrieved_info}
    Answer:
    """
    extracted_answer = call_llm(prompt, system_instruction)
    print(f"Extracted answer: {extracted_answer}") # Debugging
    return extracted_answer

def validate_answer(question, answer):
    """Validate the extracted answer against the question."""
    system_instruction = "You are a fact validator, ensuring the answer is correct and complete."
    prompt = f"""
    Validate if the answer accurately and completely answers the question.
    
    Example 1:
    Question: In what year did Charlton Publications sell Hit Parader?
    Answer: 1991
    Validation: VALID
    
    Example 2:
    Question: What is the peak brightness of the Asus ROG Phone 5s Pro in nits?
    Answer: 1200 nits
    Validation: VALID
    
    Example 3:
    Question: Who played Creon in Antigone at the Epidaurus Festival 2022?
    Answer: Vasilis Bisbikis
    Validation: VALID
    
    Question: {question}
    Answer: {answer}
    Validation:
    """
    validation_result = call_llm(prompt, system_instruction)
    print(f"Validation result: {validation_result}") # Debugging
    return validation_result

def main(question):
    """Solve questions by generating a search query, retrieving info, extracting, and validating."""
    try:
        # 1. Generate Search Query
        search_query = generate_search_query(question)
        
        # 2. Retrieve Information
        retrieved_info = retrieve_info(search_query)
        if "Error" in retrieved_info:
          return "Error retrieving information."
        
        # 3. Extract Answer
        answer = extract_answer(question, retrieved_info)
        if "Error" in answer:
          return "Error extracting answer."

        # 4. Validate Answer
        validation_result = validate_answer(question, answer)
        if "Error" in validation_result:
          return "Error validating answer."
        
        if "VALID" in validation_result:
            return answer
        else:
            return "Could not be validated."

    except Exception as e:
        return f"Error: {str(e)}"