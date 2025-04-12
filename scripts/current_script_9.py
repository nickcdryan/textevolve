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
    """Extract relevant info using LLM"""
    system_instruction = "You are an expert at extracting relevant scheduling information. Focus on participants and their blocked times."
    prompt = f"Extract participants and their blocked times from this scheduling problem: {problem}"
    return call_llm(prompt, system_instruction)

def propose_solution_llm(info):
    """Propose a solution using LLM"""
    system_instruction = "You are an expert at proposing meeting times based on participant availability."
    prompt = f"Based on this information, suggest a possible meeting time: {info}"
    return call_llm(prompt, system_instruction)

def critique_solution_llm(problem, proposed_solution):
    """Critique the solution using LLM"""
    system_instruction = "You are a solution critic who identifies potential issues with a proposed meeting time."
    prompt = f"Given the scheduling problem: {problem} and the proposed solution: {proposed_solution}, identify any potential conflicts or issues."
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting"""
    try:
        extracted_info = extract_info_llm(question)
        proposed_solution = propose_solution_llm(extracted_info)
        critique = critique_solution_llm(question, proposed_solution)

        if "conflict" in critique.lower() or "issue" in critique.lower():
            return "No suitable time found."
        else:
            return f"Proposed time: {proposed_solution}"
    except Exception as e:
        return f"Error: {str(e)}"