import os
import re
import math

def main(question):
    """
    Solve the question using a multi-stage LLM approach. Includes multiple examples and handles potential errors.
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
    """Analyzes the question to identify its type and keywords. Includes multiple examples for robust analysis."""
    system_instruction = "You are an expert at analyzing questions to determine their type and keywords."
    prompt = f"""
    Analyze the following question and identify its type (e.g., fact extraction, calculation, comparison) and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "player"]}}

    Example 2:
    Question: How many running backs ran for a touchdown?
    Analysis: {{"type": "counting", "keywords": ["running backs", "touchdown", "number"]}}
    
    Example 3:
    Question: Which player kicked the only field goal of the game?
    Analysis: {{"type": "fact extraction", "keywords": ["player", "field goal", "kicked"]}}

    Example 4:
    Question: How many percent were not english?
    Analysis: {{"type": "calculation", "keywords": ["percent", "not english"]}}

    Question: {question}
    Analysis:
    """
    return call_llm(prompt, system_instruction)

def extract_relevant_passage(question, question_analysis):
    """Extracts the relevant passage from the question based on keywords. Robust extraction with diverse examples."""
    system_instruction = "You are an expert at extracting relevant passages from text."
    prompt = f"""
    Extract the relevant passage from the following text based on the question and keywords.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["final touchdown", "caught", "player"]}}
    Text: PASSAGE: After a tough loss at home, the Browns traveled to take on the Packers. ... The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    
    Example 2:
    Question: How many running backs ran for a touchdown?
    Keywords: {{"type": "counting", "keywords": ["running backs", "touchdown", "number"]}}
    Text: PASSAGE: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. The Lions would respond with kicker Jason Hanson getting a 53-yard field goal. The Titans would answer with Johnson getting a 58-yard TD run, along with DE Dave Ball returning an interception 15 yards for a touchdown. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Keywords: {{"type": "fact extraction", "keywords": ["player", "field goal", "kicked"]}}
    Text: PASSAGE: Game SummaryComing off their Thanksgiving road win over the Falcons, the Colts went home for a Week 13 AFC South rematch with the Jacksonville Jaguars.  ... In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.

    Example 4:
    Question: How many percent were not english?
    Keywords: {{"type": "calculation", "keywords": ["percent", "not english"]}}
    Text: PASSAGE: As of the 2010 United States Census, there were 146,551 people, 51,214 households, and 38,614 families residing in the county. The population density was . There were 54,963 housing units at an average density of . The racial makeup of the county was 50.3% white, 41.0% black or African American, 3.0% Asian, 0.7% American Indian, 0.1% Pacific islander, 1.3% from other races, and 3.7% from two or more races. Those of Hispanic or Latino origin made up 4.3% of the population. In terms of ancestry, 12.6% were Germans, 10.8% were Irish people, 8.7% were English people, 6.3% were Americans, and 5.1% were Italians.
    Passage: As of the 2010 United States Census, there were 146,551 people, 51,214 households, and 38,614 families residing in the county. The racial makeup of the county was 50.3% white, 41.0% black or African American, 3.0% Asian, 0.7% American Indian, 0.1% Pacific islander, 1.3% from other races, and 3.7% from two or more races. In terms of ancestry, 12.6% were Germans, 10.8% were Irish people, 8.7% were English people, 6.3% were Americans, and 5.1% were Italians.

    Question: {question}
    Keywords: {question_analysis}
    Text: {question}
    Passage:
    """
    return call_llm(prompt, system_instruction)

def generate_answer(question, relevant_passage, question_analysis):
    """Generates the answer based on the question, relevant passage, and question type. Rich examples."""
    system_instruction = "You are an expert at generating answers to questions based on provided text. You pay close attention to the question type."
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

    Example 4:
    Question: How many percent were not english?
    Passage: As of the 2010 United States Census, there were 146,551 people, 51,214 households, and 38,614 families residing in the county. The population density was . There were 54,963 housing units at an average density of . The racial makeup of the county was 50.3% white, 41.0% black or African American, 3.0% Asian, 0.7% American Indian, 0.1% Pacific islander, 1.3% from other races, and 3.7% from two or more races. In terms of ancestry, 12.6% were Germans, 10.8% were Irish people, 8.7% were English people, 6.3% were Americans, and 5.1% were Italians.
    Answer: 91.3

    Question: {question}
    Passage: {relevant_passage}
    Question Analysis: {question_analysis}
    Answer:
    """
    return call_llm(prompt, system_instruction)

def verify_answer(question, answer, relevant_passage):
    """Verifies the generated answer. Includes calculation ability and many examples. Addresses identified weaknesses."""
    system_instruction = "You are an expert at verifying answers to questions, including performing calculations and checking for units. Provide the correct answer with reasoning."
    prompt = f"""
    Verify the following answer to the question based on the relevant passage. You should identify if a calculation is needed. If the answer requires a calculation, you MUST perform the calculation and give the correct answer. Return the correct answer with reasoning.

    Example 1:
    Question: Who caught the final touchdown of the game?
    Answer: Jarrett Boykin
    Passage: The Packers would later on seal the game when Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score 31-13.
    Verification: Jarrett Boykin is correct because the passage states 'Rodgers found Jarrett Boykin on a 20-yard pass for the eventual final score'.

    Example 2:
    Question: How many running backs ran for a touchdown?
    Answer: 2
    Passage: In the first quarter, Tennessee drew first blood as rookie RB Chris Johnson got a 6-yard TD run. In the second quarter, Tennessee increased their lead with RB LenDale White getting a 6-yard and a 2-yard TD run.
    Verification: 2 is correct. Chris Johnson and LenDale White are two running backs who ran for a touchdown.

    Example 3:
    Question: Which player kicked the only field goal of the game?
    Answer: Josh Scobee
    Passage: In the fourth quarter, the Jaguars drew closer as kicker Josh Scobee nailed a 47-yard field goal.
    Verification: Josh Scobee is the correct answer because the passage mentions Josh Scobee kicked a field goal.

    Example 4:
    Question: How many percent were not english?
    Answer: 91.3
    Passage: As of the 2010 United States Census, there were 146,551 people... In terms of ancestry, 12.6% were Germans, 10.8% were Irish people, 8.7% were English people...
    Verification: 91.3 is correct because 100 - 8.7 = 91.3.

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