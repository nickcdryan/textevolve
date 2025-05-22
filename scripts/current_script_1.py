import os
from google import genai
from google.genai import types
import re

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

def extract_entities_and_relationships(question):
    """Extract key entities and relationships from the question using LLM with multiple examples."""
    system_instruction = "You are an expert information extractor specializing in entities and relationships."

    prompt = f"""
    Extract the key entities and relationships from the following question. Provide the entities and relationships in plain text.

    Example 1:
    Question: What is the name of the man who purchased Belmont on St. Saviour's Hill, Jersey, UK, in 1822?
    Entities: man, Belmont, St. Saviour's Hill, Jersey, UK
    Relationships: purchased Belmont on St. Saviour's Hill in 1822

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Entities: corners, Barcelona, Champions League semi-final, Milan, April 27, 2006
    Relationships: Barcelona took corners in match against Milan on April 27, 2006

    Example 3:
    Question: Specify the day, month, and year in which Activision Blizzard announced the upcoming establishment of a new esports division.
    Entities: day, month, year, Activision Blizzard, esports division
    Relationships: Activision Blizzard announced esports division

    Question: {question}
    Entities and Relationships:
    """
    return call_llm(prompt, system_instruction)

def generate_search_query(entities_and_relationships, question):
    """Generate a search query from extracted entities and relationships using LLM with multiple examples."""
    system_instruction = "You are an expert query generator."

    prompt = f"""
    Generate a search query from the extracted entities and relationships. The search query should be optimized for finding the answer to the question.

    Example 1:
    Question: What is the name of the man who purchased Belmont on St. Saviour's Hill, Jersey, UK, in 1822?
    Entities and Relationships: man, Belmont, St. Saviour's Hill, Jersey, UK; purchased Belmont on St. Saviour's Hill in 1822
    Search Query: "man purchased Belmont St. Saviour's Hill Jersey 1822"

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Entities and Relationships: corners, Barcelona, Champions League semi-final, Milan, April 27, 2006; Barcelona took corners in match against Milan on April 27, 2006
    Search Query: "Barcelona Milan Champions League semi-final corners April 27 2006"

    Example 3:
    Question: Specify the day, month, and year in which Activision Blizzard announced the upcoming establishment of a new esports division.
    Entities and Relationships: day, month, year, Activision Blizzard, esports division; Activision Blizzard announced esports division
    Search Query: "Activision Blizzard esports division announcement date"

    Question: {question}
    Entities and Relationships: {entities_and_relationships}
    Search Query:
    """
    return call_llm(prompt, system_instruction)

def retrieve_information(search_query):
    """Simulate information retrieval from a search engine."""
    # In a real implementation, this would call an actual search API
    system_instruction = "You are a search engine that provides concise information."
    return call_llm(f"Provide information about: {search_query}", system_instruction)

def generate_answer(question, retrieved_information):
    """Generate the answer from the retrieved information using LLM with multiple examples."""
    system_instruction = "You are an expert answer generator."

    prompt = f"""
    Generate an answer to the question using the retrieved information.

    Example 1:
    Question: What is the name of the man who purchased Belmont on St. Saviour's Hill, Jersey, UK, in 1822?
    Retrieved Information: Sir Colin Halkett purchased Belmont on St. Saviour's Hill, Jersey, UK, in 1822.
    Answer: Sir Colin Halkett

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Retrieved Information: Barcelona took 3 corners in the Champions League semi-final match between Barcelona and Milan on April 27, 2006.
    Answer: 3

    Example 3:
    Question: Specify the day, month, and year in which Activision Blizzard announced the upcoming establishment of a new esports division.
    Retrieved Information: Activision Blizzard announced the upcoming establishment of a new esports division on 21 of October of 2015.
    Answer: 21 of October of 2015

    Question: {question}
    Retrieved Information: {retrieved_information}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def validate_answer(question, answer):
    """Validate the generated answer against the question using LLM with multiple examples."""
    system_instruction = "You are an expert answer validator."

    prompt = f"""
    Validate if the answer correctly answers the question. If there are issues respond with INVALID: [explain issues], else respond with VALID: [brief explanation]

    Example 1:
    Question: What is the name of the man who purchased Belmont on St. Saviour's Hill, Jersey, UK, in 1822?
    Answer: Sir Colin Halkett
    Validation: VALID: The answer provides the name of the man.

    Example 2:
    Question: How many corners did Barcelona take in the Champions League semi-final match between Barcelona and Milan on April 27, 2006?
    Answer: 3
    Validation: VALID: The answer provides the number of corners.

    Example 3:
    Question: Specify the day, month, and year in which Activision Blizzard announced the upcoming establishment of a new esports division.
    Answer: 21 of October of 2015
    Validation: VALID: The answer provides the day, month, and year.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """
    Solve a question using LLM by extracting entities and relationships, generating a search query,
    retrieving information, generating an answer, and validating the answer.
    """
    try:
        # Step 1: Extract entities and relationships
        entities_and_relationships = extract_entities_and_relationships(question)

        # Step 2: Generate a search query
        search_query = generate_search_query(entities_and_relationships, question)

        # Step 3: Retrieve information
        retrieved_information = retrieve_information(search_query)

        # Step 4: Generate an answer
        answer = generate_answer(question, retrieved_information)

        # Step 5: Validate the answer
        validation_result = validate_answer(question, answer)

        if "VALID:" in validation_result:
            return answer
        else:
            return f"Error: {validation_result}"

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"