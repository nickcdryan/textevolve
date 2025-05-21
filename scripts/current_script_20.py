import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach with enhanced error handling and reasoning.
    """
    try:
        # Step 1: Identify question type and keywords
        question_analysis = analyze_question(question)
        if "Error" in question_analysis:
            return f"Error analyzing question: {question_analysis}"

        # Step 2: Extract relevant passage using identified keywords
        relevant_passage = extract_relevant_passage(question, question_analysis)
        if "Error" in relevant_passage:
            return f"Error extracting passage: {relevant_passage}"

        # Step 3: Generate answer using extracted passage and question type
        answer = generate_answer(question, relevant_passage, question_analysis)
        if "Error" in answer:
            return f"Error generating answer: {answer}"

        # Step 4: Verify answer
        verified_answer = verify_answer(question, answer, relevant_passage, question_analysis)
        if "Error" in verified_answer:
            return f"Error verifying answer: {verified_answer}"
        
        return verified_answer

    except Exception as e:
        return f"General Error: {str(e)}"

def analyze_question(question):
    """Analyzes the question to identify its type and keywords. Includes multiple examples."""
    system_instruction = "You are an expert at analyzing questions to determine their type and keywords."
    prompt = f"""
    Analyze the following question and identify its type (e.g., fact extraction, calculation, comparison) and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught"]}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown"], "requires_calculation": "yes"}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords. Includes multiple examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract the relevant passage from the following text based on the question and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {question_analysis}
    Text: PASSAGE: After a tough loss at home, the Browns traveled to take on the Packers. ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {question_analysis}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run, along with DE Dave Ball returning an interception 15 yards for a touchdown. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Keywords: {question_analysis}
    Text: PASSAGE: Game SummaryComing off their Thanksgiving road win over the Falcons, the Colts went home for a Week 13 AFC South rematch with the Jacksonville Jaguars.  ... In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type. Includes multiple examples."""
    system_instruction = "You are an expert at generating answers to questions based on provided text."
    prompt = f"""
    Generate the answer to the question based on the relevant passage and question type.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Answer: Jarrett Boykin

    Example 2:
    Question: How many running backs ran for a touchdown?
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run, along with DE Dave Ball returning an interception 15 yards for a touchdown. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Answer: 2
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Answer: Josh Scobee

    Question: {question}
    Passage: {relevant_passage}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage, question_analysis):
    """Verifies the generated answer. Includes multiple examples. Now with calculation abilities."""
    system_instruction = "You are an expert at verifying answers to questions, including performing calculations when needed."
    requires_calculation = "yes" in question_analysis.lower()

    if requires_calculation:
      # If calculation is needed, prompt LLM to perform calculation
      prompt = f"""
      The question requires a calculation. Please perform the calculation based on the provided passage and return the result.

      Example 1:
      Question: How many points did the Packers score?
      Answer: There was a TD by Rodgers, a TD by Boykin and a field goal. TDs are 7 points and the field goal is 3.
      Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
      Calculation: 7 + 7 + 3 = 17
      Verification: 17
      
      Question: {question}
      Answer: {answer}
      Passage: {relevant_passage}
      Calculation:
      """
      return call_llm(prompt, system_instruction)
    else:
      # If no calculation is needed, just return the original answer
      prompt = f"""
      Verify the following answer to the question based on the relevant passage. Return the answer if it is correct. Return the correct answer if it is incorrect.

      Example 1:
      Question: Who caught the final touchdown of the game?
      Answer: Jarrett Boykin
      Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
      Verification: Jarrett Boykin
      
      Question: {question}
      Answer: {answer}
      Passage: {relevant_passage}
      Verification:
      """
      return call_llm(prompt, system_instruction)

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