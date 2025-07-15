import os
from abc import ABC, abstractmethod

from google import genai
from google.genai import types
from openai import OpenAI


class LLMClient(ABC):

    @property
    def provider(self):
        raise NotImplementedError

    @abstractmethod
    def call_llm(self, prompt: str, system_instruction: str = "") -> str:
        pass

class GeminiClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"))
            print("Gemini API client initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini API client: {e}")
            print("Make sure to set the GEMINI_API_KEY environment variable")
            raise

    @property
    def provider(self):
        return "Gemini"

    def call_llm(self, prompt: str, system_instruction: str = "") -> str:
        """Call the Gemini LLM with a prompt and return the response"""
        try:
            # Use provided system instruction or default to the loaded system prompt
            sys_instruction = system_instruction if system_instruction is not None else ""

            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction),
                contents=prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"



class OpenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")
        self.client = OpenAI(api_key=api_key, organization=org_id) if org_id else OpenAI(api_key=api_key)
    
    @property
    def provider(self):
        return "OpenAI"

    def call_llm(self, prompt: str, system_instruction: str = "") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error: {str(e)}"