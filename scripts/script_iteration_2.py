import os
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

def main(question, supporting_documents):
    """
    This script tests a 'knowledge graph' approach.
    It extracts entities and relationships from documents, then uses these to answer the question.
    """

    # Hypothesis: Building a simple knowledge graph can help with multi-hop reasoning.

    # Step 1: Extract knowledge (entities and relationships) from each document.
    knowledge = []
    for i, doc in enumerate(supporting_documents):
        knowledge_extraction_result = extract_knowledge(doc, i, question)
        if knowledge_extraction_result.get("is_valid"):
            knowledge.extend(knowledge_extraction_result["knowledge"])
        else:
            print(f"Knowledge extraction failed for document {i}: {knowledge_extraction_result.get('validation_feedback')}")
            return f"Knowledge extraction failed for document {i}"

    # Step 2: Reason over the extracted knowledge to answer the question.
    answer = reason_over_knowledge(question, knowledge)
    return answer

def extract_knowledge(document, doc_id, question, max_attempts=3):
    """Extracts entities and relationships from a document."""
    system_instruction = "You are a knowledge extraction expert."

    for attempt in range(max_attempts):
        extraction_prompt = f"""
        Extract entities and relationships from this document relevant to the question: {question}.
        Output a list of dictionaries, where each dictionary represents a fact.

        Example:
        Document: "Oasis was formed in 1991 by Liam Gallagher and Noel Gallagher."
        Output:
        [
            {{"entity1": "Oasis", "relation": "formed", "entity2": "Liam Gallagher", "year": "1991"}},
            {{"entity1": "Oasis", "relation": "formed", "entity2": "Noel Gallagher", "year": "1991"}}
        ]

        Document: {document}
        Output:
        """

        extracted_knowledge = call_llm(extraction_prompt, system_instruction)

        # Step 3: Validation step - is the knowledge in the correct format?
        validation_prompt = f"""
        Verify that the extracted knowledge from document {doc_id} is valid and in the correct format.

        Extracted Knowledge: {extracted_knowledge}

        Respond with "VALID" if the format is correct, otherwise "INVALID: [reason]"

        Example 1:
        Extracted Knowledge: [{{"entity1": "Oasis", "relation": "formed", "entity2": "Liam Gallagher", "year": "1991"}} ]
        Validation: VALID

        Example 2:
        Extracted Knowledge: Oasis was formed by Liam.
        Validation: INVALID: Knowledge should be a list of dictionaries.
        """

        validation_result = call_llm(validation_prompt, system_instruction)

        if "VALID" in validation_result:
            try:
                #Attempt conversion to data to verify. The call_llm JSON output will be plain text, so we will try to convert it to code but if it fails, then it will go to except
                data = eval(extracted_knowledge)
                return {"is_valid": True, "knowledge": data}
            except:
                return {"is_valid": False, "validation_feedback": "Could not be converted using eval()"}
        else:
            print(f"Knowledge extraction validation failed for doc {doc_id}, attempt {attempt+1}: {validation_result}")
            if attempt < max_attempts - 1:
                continue
            else:
                return {"is_valid": False, "knowledge": [], "validation_feedback": validation_result}
    return {"is_valid": False, "knowledge": [], "validation_feedback": "Failed after multiple attempts."}

def reason_over_knowledge(question, knowledge):
    """Reasons over the extracted knowledge to answer the question."""
    system_instruction = "You are an expert question answering system."

    reasoning_prompt = f"""
    Based on this extracted knowledge, answer the question: {question}.

    Extracted Knowledge:
    {knowledge}

    Example:
    Question: Who founded Oasis?
    Extracted Knowledge:
    [
        {{"entity1": "Oasis", "relation": "formed", "entity2": "Liam Gallagher", "year": "1991"}},
        {{"entity1": "Oasis", "relation": "formed", "entity2": "Noel Gallagher", "year": "1991"}}
    ]
    Answer: Liam Gallagher and Noel Gallagher

    Question: {question}
    Answer:
    """

    answer = call_llm(reasoning_prompt, system_instruction)
    return answer