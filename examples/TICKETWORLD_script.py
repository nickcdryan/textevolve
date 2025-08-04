import os
import json
from openai import OpenAI

def call_llm(prompt, system_instruction=None):
    """
    Call the OpenAI API with the given prompt and system instruction.
    """
    # Set your API key (keep this safe and secure)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Call the chat completion endpoint
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
    )

    # Return the response content
    return response.choices[0].message.content

def main(question):
    """
    TicketWorld Customer Service Resolution System
    
    This script processes customer service emails and generates comprehensive
    resolution plans following company policies and database information.
    """
    
    system_instruction = """
    You are an expert customer service resolution specialist. You have access to:
    - Customer database with order history and customer information
    - Company policy documents with specific policy IDs
    - Product information and warranty details
    
    Your task is to analyze customer emails and create detailed resolution plans
    that follow company policies exactly and provide appropriate actions.
    
    Always:
    1. Look up customer information first
    2. Reference specific policy IDs in your reasoning
    3. Calculate exact monetary values when applicable
    4. Provide detailed step-by-step resolution actions
    5. Determine appropriate priority and escalation needs
    """
    
    # Process the customer service ticket
    prompt = f"""
    {question}
    
    Based on the customer email above, please generate a comprehensive resolution plan.
    Follow the exact schema provided in the instructions and ensure all required fields are included.
    
    Remember to:
    - Search for the customer in the database by email
    - Look up order information if an order is referenced
    - Find all relevant policies from company_policy.txt
    - Provide specific policy citations (e.g., POL-WARRANTY-001)
    - Calculate exact dollar amounts for any monetary actions
    - Include detailed reasoning for your decisions
    - Set appropriate priority level and escalation status
    """
    
    try:
        # Call the LLM to generate the resolution plan
        response = call_llm(prompt, system_instruction)
        
        # Try to parse the response as JSON to validate format
        try:
            resolution_plan = json.loads(response)
            # If it parses successfully, return the formatted JSON
            return json.dumps(resolution_plan, indent=2)
        except json.JSONDecodeError:
            # If it's not valid JSON, return the raw response
            return response
            
    except Exception as e:
        return f"Error processing customer service ticket: {str(e)}"

if __name__ == "__main__":
    # Example usage - this would normally be called by the agent system
    sample_question = """
    Customer Email: test@example.com
    Subject: Return request
    Timestamp: 2025-01-01T10:00:00
    
    Message Body:
    I would like to return my recent order. The product doesn't meet my expectations.
    
    Please provide a resolution plan for this customer service ticket following the schema and instructions above.
    """
    
    result = main(sample_question)
    print(result) 