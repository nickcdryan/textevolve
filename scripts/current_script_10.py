import os
import json

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def extract_info_llm(problem):
    system_instruction = "You are a scheduler. Extract participants, duration, and constraints from the problem."
    prompt = f"Extract information to schedule the meeting from: {problem}"
    return call_llm(prompt, system_instruction)

def propose_solution_llm(info):
    system_instruction = "You are a helpful assistant, propose a solution."
    prompt = f"Based on this information, propose a meeting time: {info}"
    return call_llm(prompt, system_instruction)

def critique_solution_llm(info, proposed_solution):
    system_instruction = "You are a solution critic. Analyze if the solution is valid or not. If invalid, explain reason."
    prompt = f"Information:\n{info}\nProposed Solution:\n{proposed_solution}\nCritique the proposed solution."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting"""
    try:
        info = extract_info_llm(question)
        proposed_solution = propose_solution_llm(info)
        critique = critique_solution_llm(info, proposed_solution)

        if "invalid" in critique.lower() or "not valid" in critique.lower() or "doesn't work" in critique.lower():
            return "No valid solution found."
        else:
            return proposed_solution
    except Exception as e:
        return f"An error occurred: {str(e)}"