import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced question analysis and verification.
    """
    try:
        # Step 1: Identify question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return "Error analyzing question"

        # Step 2: Extract relevant passage using identified keywords
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return "Error extracting passage"

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return "Error generating answer"

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage)
        if "Error" in verified_answer:
            return "Error verifying answer"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords, adding more examples and detail."""
    system_instruction = "You are an expert at analyzing questions."
    prompt = f"""
    Analyze the question to identify its type and keywords for targeted information extraction.
    Explicitly identify if calculation is needed, and what type.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"], "calculation_needed": "no"}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "calculation_needed": "yes", "calculation_type": "count"}}

    Example 3:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "Jason Hanson", "touchdown", "field goal"], "calculation_needed": "yes", "calculation_type": "addition"}}
    
    Example 4:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"], "calculation_needed": "no"}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts relevant passage, also extracting the question keywords. Adding more examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract the relevant passage, considering question type and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}}
    Text: PASSAGE: ... Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"]}}
    Text: PASSAGE: ...Chris Johnson got a 6-yard TD run. ...LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: Chris Johnson got a 6-yard TD run. LenDale White getting a 6-yard and a 2-yard TD run.

    Example 3:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "Jason Hanson", "touchdown", "field goal"], "calculation_needed": "yes", "calculation_type": "addition"}}
    Text: PASSAGE: Chris Johnson got a 6-yard TD run. ...Jason Hanson getting a 53-yard field goal.
    Passage: Chris Johnson got a 6-yard TD run. Jason Hanson getting a 53-yard field goal.
    
    Question: {question}
    Analysis: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates answer, factoring in question analysis for more targeted responses. More examples."""
    system_instruction = "You are an expert at generating concise and accurate answers."
    prompt = f"""
    Generate the answer. Use the question type, extracted passage to guide your response.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: Chris Johnson got a 6-yard TD run. LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: 2
    
    Example 3:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Passage: Chris Johnson got a 6-yard TD run. Jason Hanson getting a 53-yard field goal.
    Analysis: {{"type": "calculation", "keywords": ["Chris Johnson", "Jason Hanson", "touchdown", "field goal"], "calculation_needed": "yes", "calculation_type": "addition"}}
    Answer: 59
    
    Question: {question}
    Passage: {relevant_passage}
    Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the answer, correcting if necessary. Uses examples that emphasize different types of reasoning."""
    system_instruction = "You are an expert at verifying answers and performing calculations to arrive at the right result."
    prompt = f"""
    Verify the answer. If incorrect, provide the correct answer based on the passage. If a calculation is needed, perform it and provide the result.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: Chris Johnson got a 6-yard TD run. LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: 2

    Example 3:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: It's Chris Johnson and Jason Hanson.
    Passage: Chris Johnson got a 6-yard TD run. Jason Hanson getting a 53-yard field goal.
    Verification: 59

    Question: {question}
    Answer: {answer}
    Passage: {relevant_passage}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template."""
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