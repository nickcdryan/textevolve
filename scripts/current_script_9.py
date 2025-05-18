import os
import re
import math

def main(question):
    """
    This script uses a radically different approach: a "Reading Comprehension Expert" agent.
    The agent will carefully review the passage and question, then engage in a self-debate
    to ensure a correct and comprehensive answer. This avoids brittle decomposition and
    focuses on direct reasoning. The hypothesis is that a more holistic understanding
    will improve accuracy. Verification checks after the debate will help validate this.
    """
    try:
        reading_comprehension_expert = ReadingComprehensionExpert()
        answer = reading_comprehension_expert.answer_question(question)
        return answer
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

class ReadingComprehensionExpert:
    """
    A class that embodies the Reading Comprehension Expert.
    """
    def __init__(self):
        self.system_instruction = """You are a Reading Comprehension Expert with a deep understanding of passages.
        You will analyze the passage and question and provide a well-reasoned answer."""

    def answer_question(self, question, max_attempts=3):
        """
        Answers the question using a self-debate strategy for improved accuracy.
        """
        initial_analysis = self._analyze_question(question)
        debate_result = self._conduct_self_debate(question, initial_analysis)

        for attempt in range(max_attempts):
            # Check the debate result for validity
            verification_prompt = f"""
            Verify the result of the self-debate is a valid and well reasoned answer to the original problem.

            Original Question: {question}

            Self-Debate result: {debate_result}

            Example:
            Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
            Self-Debate result: The expert debater has arrived to the conclusion that the final answer is 59.
            Valid
            """
            validation_result = call_llm(verification_prompt, self.system_instruction)

            if "valid" in validation_result.lower():
                return debate_result
            else:
                print(f"Result failed to validate")

        return "The expert debater did not arrive at a conclusive and valid answer. Please check your source data, reasoning, and the validity of the answer."

    def _analyze_question(self, question):
        """Analyzes the question and extracts key information."""
        analysis_prompt = f"""
        Analyze the question and passage to understand what's being asked.

        Question: {question}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Analysis: The question requires finding two numbers and adding them.

        Analysis:
        """
        return call_llm(analysis_prompt, self.system_instruction)

    def _conduct_self_debate(self, question, initial_analysis):
        """Conducts a self-debate to refine the answer."""
        debate_prompt = f"""
        Engage in a self-debate to ensure the most accurate and comprehensive answer.

        Question: {question}
        Initial Analysis: {initial_analysis}

        Example:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Initial Analysis: The question requires finding two numbers and adding them.
        Expert 1: The passage contains data on Johnson and Hanson
        Expert 2: Let us verify those numbers and make sure the math works
        Conclusion: The expert debater has arrived to the conclusion that the final answer is 59.

        Debate:
        """
        return call_llm(debate_prompt, self.system_instruction + " You are an expert debater.")

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types
        import os

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