import os
import re

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def schedule_meeting(question):
    """Schedules a meeting based on the given constraints using LLM."""

    # Agent 1: Constraint Extractor and Availability Summarizer
    def extract_and_summarize(problem):
        system_instruction = "You are an expert at extracting meeting constraints and summarizing participant availability. Provide the constraints and a concise summary of each participant's availability. List unavailable slots."
        prompt = f"Extract the meeting constraints and summarize availability from: {problem}"
        return call_llm(prompt, system_instruction)

    # Agent 2: Candidate Time Generator and Conflict Resolver
    def generate_and_resolve(availability_summary, duration="half an hour"):
        system_instruction = "You are a meeting scheduler. Given availability summaries, generate three candidate meeting times and resolve any conflicts. Explicitly state why each is valid. Provide the times in the format '[Day], [Start Time] - [End Time]'."
        prompt = f"Generate three candidate meeting times based on this availability summary: {availability_summary}. Meeting duration is {duration}."
        return call_llm(prompt, system_instruction)

    # Agent 3: Solution Selector and Output Formatter
    def select_and_format(candidate_times):
        system_instruction = "You are a final decision-maker. From the candidate times, select the best one based on earliest availability and format the output string. Provide the time in the format '[Day], [Start Time] - [End Time]'."
        prompt = f"Select the best meeting time from these candidates: {candidate_times}. Format the answer as: 'Here is the proposed time: [Day], [Start Time] - [End Time]'"
        return call_llm(prompt, system_instruction)

    # Chain-of-thought execution
    try:
        availability_summary = extract_and_summarize(question)
        candidate_times = generate_and_resolve(availability_summary)
        final_answer = select_and_format(candidate_times)
        return final_answer
    except Exception as e:
        return f"Error in scheduling process: {str(e)}"

def main(question):
    """Main function to schedule the meeting."""
    return schedule_meeting(question)