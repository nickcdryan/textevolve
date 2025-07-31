#!/usr/bin/env python
"""
Synthetic Customer Service Data Generator
Generates realistic customer service tickets, databases, and policy documents for testing LLM systems.
"""

import json
import random
import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# # You'll need to implement this function with your LLM API details
# def call_llm(prompt: str, system_instruction: str = None) -> str:
#     """
#     Call your LLM API here. Replace with your implementation.
    
#     Args:
#         prompt: The user prompt
#         system_instruction: Optional system instruction
    
#     Returns:
#         LLM response as string
#     """
#     pass

# def call_llm(prompt, system_instruction=None):
#     """Call the Gemini LLM with a prompt and return the response."""
#     try:
#         import google.generativeai as genai
#         import os

#         # Configure the API
#         genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

#         # Create the model
#         model = genai.GenerativeModel("gemini-2.0-flash")

#         # Combine system instruction with prompt if provided
#         if system_instruction:
#             full_prompt = f"{system_instruction}\n\n{prompt}"
#         else:
#             full_prompt = prompt

#         # Generate response
#         response = model.generate_content(full_prompt)
#         return response.text
#     except Exception as e:
#         print(f"Error calling Gemini API: {str(e)}")
#         return f"Error: {str(e)}"
    

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import os
        from google import genai
        from google.genai import types

        # Configure the client with API key
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                seed=42,
                temperature=0.7,
                max_output_tokens=8192,  # Increased token limit
            ),
        )
        
        if response.text:
            return response.text.strip()
        else:
            print("Warning: Empty response from Gemini API")
            return ""
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

class CustomerServiceDataGenerator:
    """Generates comprehensive synthetic customer service data"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.customers = {}
        self.orders = {}
        self.company_policy = ""
        self.tickets = []
        
    def generate_complete_dataset(self, 
                                num_customers: int = 100,
                                num_tickets: int = 200) -> Dict[str, Any]:
        """Generate complete dataset with all components"""
        
        print("Generating company policy document...")
        self.company_policy = self.generate_company_policy()
        
        print(f"Generating {num_customers} customers and their orders...")
        self.customers, self.orders = self.generate_customer_database(num_customers)
        
        print(f"Generating {num_tickets} support tickets...")
        self.tickets = self.generate_tickets(num_tickets)
        
        print("Generating resolution plans for all tickets...")
        tickets_with_resolutions = self.generate_resolution_plans(self.tickets)
        
        return {
            "company_policy": self.company_policy,
            "customers": self.customers,
            "orders": self.orders,
            "tickets": tickets_with_resolutions
        }
    
    def generate_company_policy(self) -> str:
        """Generate comprehensive company policy document"""
        
        system_instruction = """You are a corporate policy writer creating comprehensive customer service guidelines for an e-commerce company called "TechFlow Solutions". Create realistic, detailed policies that customer service agents would actually use."""
        
        prompt = """
Create a comprehensive customer service policy document for TechFlow Solutions, an e-commerce company selling electronics and accessories. The document should be approximately 2000 words and include both relevant policies and some irrelevant sections (to test information retrieval).

REQUIRED POLICY SECTIONS (these will be referenced in tickets):

1. REFUND POLICIES
   - Standard 30-day return window
   - Conditions for exceptions (defective products, shipping errors)
   - Restocking fees for certain categories
   - Processing timeframes

2. SHIPPING ERROR COMPENSATION  
   - Full refund policy for wrong items shipped
   - Additional 20% credit for shipping errors
   - Rush replacement procedures
   - Investigation process for repeated errors

3. CUSTOMER TIER BENEFITS
   - Standard customers: basic support
   - Premium customers: priority support, extended returns
   - VIP customers: dedicated support, manager escalation for issues
   - Tier qualification criteria

4. AUTHORIZATION LEVELS
   - Agent level: up to $50 in credits/refunds without approval
   - Supervisor level: up to $200 in credits/refunds  
   - Manager level: up to $500 in credits/refunds
   - VP approval required: over $500

5. ESCALATION TRIGGERS
   - When to escalate to supervisor vs manager
   - VIP customer special handling
   - Repeat issue protocols
   - Complaint severity levels

6. COMMUNICATION GUIDELINES
   - Tone requirements for different situations
   - Apology protocols
   - Response timeframes by priority level

IRRELEVANT SECTIONS TO INCLUDE (add realistic noise):
- Company history and founding story
- Office locations and hours
- Employee handbook excerpts  
- IT security policies
- Marketing guidelines
- Vendor relationships
- DO NOT actually say in the document that these sections are irrelevant, they should appear as if they are relevant to the policy.

Format as a professional policy document with section numbers, clear headers, and specific procedures. Include exact dollar amounts, timeframes, and step-by-step processes that can be referenced and followed.
"""
        
        return call_llm(prompt, system_instruction)
    
    def generate_customer_database(self, num_customers: int) -> tuple[Dict, Dict]:
        """Generate customer and order databases in manageable batches"""
        
        all_customers = {}
        all_orders = {}
        
        # Generate in smaller batches to avoid API limits and JSON parsing issues
        batch_size = min(5, num_customers)  # Process 5 customers at a time
        num_batches = (num_customers + batch_size - 1) // batch_size
        
        print(f"Generating {num_customers} customers in {num_batches} batches of {batch_size}...")
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            customers_in_batch = min(batch_size, num_customers - start_idx)
            
            print(f"Processing batch {batch_num + 1}/{num_batches} ({customers_in_batch} customers)...")
            
            try:
                customers, orders = self._generate_customer_batch(customers_in_batch, start_idx)
                all_customers.update(customers)
                all_orders.update(orders)
                print(f"Successfully generated batch {batch_num + 1}")
            except Exception as e:
                print(f"Error in batch {batch_num + 1}: {e}")
                # Try to continue with next batch
                continue
        
        print(f"Generated {len(all_customers)} customers and {len(all_orders)} orders total")
        return all_customers, all_orders
    
    def _generate_customer_batch(self, num_customers: int, start_idx: int) -> tuple[Dict, Dict]:
        """Generate a small batch of customers and their orders"""
        
        system_instruction = """You are a database engineer creating realistic customer and order data for an e-commerce company. Generate varied, interconnected data with realistic relationships. Return ONLY valid JSON without any markdown formatting or additional text."""
        
        prompt = f"""
Generate exactly {num_customers} realistic customers and their associated orders for TechFlow Solutions e-commerce database.
Start customer IDs from {start_idx + 1}.

Create realistic customer profiles with:
- Mix of customer tiers (70% standard, 20% premium, 10% VIP)
- Varied purchase histories (new customers to long-term customers)
- Different geographical locations
- Realistic support ticket histories
- Some customers with missing information (no phone, old addresses, etc.)

Product categories to include in orders:
- Laptops and computers ($500-2000)
- Phone accessories ($10-100)  
- Cables and adapters ($5-50)
- Cases and protection ($15-75)
- Audio equipment ($25-300)
- Gaming accessories ($20-150)

CUSTOMER SCHEMA:
{{
    "cust_{start_idx + 1}": {{
        "email": "realistic_email@domain.com",
        "name": "Full Name",
        "account_status": "standard/premium/vip", 
        "join_date": "YYYY-MM-DD",
        "total_orders": number,
        "total_spent": dollar_amount,
        "support_history": [
            {{"date": "YYYY-MM-DD", "issue": "issue_type", "resolved": true/false, "resolution_value": amount}}
        ],
        "phone": "+1-555-XXXX",
        "address": {{
            "street": "123 Main St",
            "city": "City", 
            "state": "ST",
            "zip": "12345"
        }}
    }}
}}

ORDER SCHEMA:
{{
    "ORD_001": {{
        "customer_id": "cust_{start_idx + 1}",
        "order_date": "YYYY-MM-DD",
        "items": [
            {{"name": "Product Name", "price": amount, "sku": "SKU123", "quantity": 1}}
        ],
        "total_amount": total,
        "status": "delivered/shipped/processing/cancelled",
        "tracking_number": "1Z123456789",
        "shipped_items": ["same as items unless there's an error"]
    }}
}}

Create varied scenarios:
- Some customers with single orders, others with multiple orders
- Include some problematic orders (wrong items shipped, delays, defects)
- 15% of orders should have some kind of issue for realistic ticket generation
- Generate realistic product names and SKUs
- Include seasonal ordering patterns

Return as valid JSON with this exact structure:
{{
    "customers": {{...customer objects...}},
    "orders": {{...order objects...}}
}}

IMPORTANT: Return ONLY the JSON object. No markdown, no explanations, no additional text.
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = call_llm(prompt, system_instruction)
                
                if not response or response.startswith("Error:"):
                    raise ValueError(f"Invalid API response: {response}")
                
                # Clean up the response - remove any potential markdown or extra text
                response = response.strip()
                
                # Try to extract JSON if it's wrapped in markdown
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    # Handle generic code blocks
                    response = response.split("```")[1].split("```")[0].strip()
                
                # Remove any leading/trailing text that isn't JSON
                if response.find("{") > 0:
                    response = response[response.find("{"):]
                if response.rfind("}") < len(response) - 1:
                    response = response[:response.rfind("}") + 1]
                
                # Parse the JSON response
                data = json.loads(response)
                
                # Validate structure
                if not isinstance(data, dict):
                    raise ValueError("Response is not a JSON object")
                
                customers = data.get("customers", {})
                orders = data.get("orders", {})
                
                if not customers:
                    raise ValueError("No customers found in response")
                
                return customers, orders
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Last attempt - save response for debugging
                    with open(f"debug_response_batch_{start_idx}.txt", "w") as f:
                        f.write(f"Raw response:\n{response}\n\nError: {e}")
                    print(f"Saved debug response to debug_response_batch_{start_idx}.txt")
                    raise ValueError(f"Could not parse JSON after {max_retries} attempts")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return {}, {}
    
    def generate_tickets(self, num_tickets: int) -> List[Dict]:
        """Generate varied support tickets with different information completeness"""
        
        # Define ticket distribution
        ticket_categories = [
            ("shipping_error", 0.25),
            ("billing_dispute", 0.20), 
            ("product_defect", 0.20),
            ("refund_request", 0.15),
            ("account_issue", 0.10),
            ("technical_support", 0.10)
        ]
        
        info_completeness_levels = [
            ("complete", 0.40),      # Email + order + name
            ("partial", 0.35),       # Email + order OR email + name
            ("minimal", 0.20),       # Just email
            ("insufficient", 0.05)   # Missing/wrong info, needs followup
        ]
        
        tickets = []
        customer_ids = list(self.customers.keys())
        order_ids = list(self.orders.keys())
        
        for i in range(num_tickets):
            # Select category and completeness level
            category = self._weighted_choice(ticket_categories)
            completeness = self._weighted_choice(info_completeness_levels)
            
            # Select random customer and order for this ticket
            customer_id = random.choice(customer_ids)
            customer = self.customers[customer_id]
            
            # Find orders for this customer
            customer_orders = [oid for oid, order in self.orders.items() 
                             if order["customer_id"] == customer_id]
            order_id = random.choice(customer_orders) if customer_orders else None
            
            ticket = self._generate_single_ticket(i+1, category, completeness, customer, order_id)
            tickets.append(ticket)
        
        return tickets
    
    def _generate_single_ticket(self, ticket_num: int, category: str, 
                               completeness: str, customer: Dict, order_id: str) -> Dict:
        """Generate a single ticket with specified characteristics"""
        
        system_instruction = f"""You are creating a realistic customer support ticket for category: {category} with information completeness level: {completeness}. Make it feel like a real customer wrote it."""
        
        order = self.orders.get(order_id, {}) if order_id else {}
        
        prompt = f"""
Create a realistic customer support ticket with these specifications:

TICKET CATEGORY: {category}
INFORMATION COMPLETENESS: {completeness}
CUSTOMER INFO: {json.dumps(customer, indent=2)}
ORDER INFO: {json.dumps(order, indent=2)}

COMPLETENESS LEVEL REQUIREMENTS:
- complete: Include email, order number, and customer name
- partial: Include email and either order number OR name (but not both)
- minimal: Include only email address
- insufficient: Use wrong email or missing critical info requiring followup

EDGE CASES TO SOMETIMES INCLUDE:
- Customer emails from different address but mentions their account email in message
- Mentions order number but gets a digit wrong
- Uses nickname instead of account name
- References "recent order" without specific order number
- Forwards ticket from someone else's email

CATEGORY-SPECIFIC SCENARIOS:

shipping_error:
- Wrong item received, damaged packaging, late delivery
- References specific products and what was actually received
- Include frustration level appropriate to issue severity

billing_dispute: 
- Unexpected charges, incorrect amounts, duplicate charges
- References specific transaction amounts and dates
- May include screenshots or receipt mentions

product_defect:
- Item not working as expected, manufacturing defects
- Include details about what's wrong and usage context
- May reference warranty concerns

refund_request:
- Changed mind, item doesn't fit, found better price elsewhere
- Include purchase timeframe and reason for return
- May test policy boundaries

account_issue:
- Can't log in, forgot password, suspicious activity
- Security-related concerns, account access problems
- May lack typical order references  

technical_support:
- How-to questions, compatibility issues, setup problems
- Product usage questions, troubleshooting requests
- May be less urgent than other categories

TICKET SCHEMA:
{{
    "ticket_id": "TKT_{ticket_num:03d}",
    "date": "2024-01-{random.randint(1, 28):02d}T{random.randint(8, 17):02d}:{random.randint(0, 59):02d}:00",
    "customer_info": {{
        // Include fields based on completeness level
        "email": "customer_email@domain.com",
        "order_number": "ORD_12345",  // if applicable
        "phone": "+1-555-1234",     // sometimes
        "name": "Customer Name"      // if applicable
    }},
    "subject": "Brief subject line",
    "message": "Realistic customer message with appropriate tone and details",
    "category": "{category}",
    "complexity_level": "simple/medium/complex"
}}

Make the message sound like a real customer wrote it - include natural language, appropriate emotional tone, and realistic details. Vary writing styles (some formal, some casual, some frustrated).

Return only valid JSON.
"""
        
        response = call_llm(prompt, system_instruction)
        
        # Parse JSON response
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            # Fallback ticket if parsing fails
            return {
                "ticket_id": f"TKT_{ticket_num:03d}",
                "date": f"2024-01-{random.randint(1, 28):02d}T{random.randint(8, 17):02d}:00:00",
                "customer_info": {"email": customer["email"]},
                "subject": f"Support request - {category}",
                "message": "I need help with my recent order.",
                "category": category,
                "complexity_level": "medium"
            }
    
    def generate_resolution_plans(self, tickets: List[Dict]) -> List[Dict]:
        """Generate policy-compliant resolution plans for all tickets"""
        
        tickets_with_resolutions = []
        
        for ticket in tickets:
            resolution = self._generate_single_resolution(ticket)
            ticket["resolution_plan"] = resolution
            tickets_with_resolutions.append(ticket)
        
        return tickets_with_resolutions
    
    def _generate_single_resolution(self, ticket: Dict) -> Dict:
        """Generate a single resolution plan for a ticket"""
        
        system_instruction = """You are an expert customer service manager creating the ideal resolution plan for a support ticket. Your resolution must strictly follow company policies and be completely verifiable."""
        
        prompt = f"""
Create the ideal resolution plan for this customer support ticket that strictly follows company policy.

TICKET:
{json.dumps(ticket, indent=2)}

COMPANY POLICY (use this as your reference):
{self.company_policy[:3000]}...  // truncated for prompt length

CUSTOMER DATABASE (for lookup):
{json.dumps(dict(list(self.customers.items())[:5]), indent=2)}...  // sample

ORDER DATABASE (for lookup):
{json.dumps(dict(list(self.orders.items())[:5]), indent=2)}...  // sample

RESOLUTION PLAN SCHEMA:
{{
    "customer_lookup": {{
        "status": "found/needs_followup/unrecognized",
        "customer_id": "cust_12345" or null,
        "confidence": "high/medium/low",
        "lookup_method": "email_match/order_match/name_phone_match",
        "notes": "Any lookup complications or assumptions"
    }},
    "policy_references": [
        "specific_policy_section_referenced",
        "another_policy_section"
    ],
    "actions": [
        {{
            "type": "refund/credit/replacement/escalation/followup",
            "amount": dollar_amount,
            "reason": "policy_compliant_justification",
            "authorization_level": "agent/supervisor/manager",
            "timeline": "immediate/24h/3-5_days"
        }}
    ],
    "escalation_required": true/false,
    "escalation_reason": "reason_if_applicable",
    "response_tone": "professional/apologetic/empathetic/firm",
    "priority": "low/medium/high/urgent",
    "total_resolution_value": total_dollar_amount,
    "policy_compliance": {{
        "within_authorization": true/false,
        "follows_procedures": true/false,
        "notes": "any_compliance_considerations"
    }},
    "followup_required": {{
        "needed": true/false,
        "reason": "insufficient_info/verification_needed/etc",
        "questions": ["What specific questions to ask"]
    }}
}}

REQUIREMENTS:
1. Customer lookup must be realistic based on provided info
2. All actions must reference specific policy sections
3. Authorization levels must match policy limits
4. Resolution value must be justified by policy
5. Handle edge cases (wrong emails, missing info) appropriately
6. Include followup requests when information is insufficient

EDGE CASE SCENARIOS:
- If email not found but order mentioned, try order lookup
- If customer mentions account email different from sender, note this
- If insufficient info, create followup request rather than guessing
- Apply VIP benefits only if customer status verified
- Escalate repeat issues per policy requirements

Return only valid JSON that strictly follows the schema.
"""
        
        response = call_llm(prompt, system_instruction)
        
        # Parse JSON response
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            # Fallback resolution if parsing fails
            return {
                "customer_lookup": {"status": "needs_followup", "confidence": "low"},
                "policy_references": ["standard_support_procedure"],
                "actions": [{"type": "followup", "reason": "need_more_information"}],
                "escalation_required": False,
                "response_tone": "professional",
                "priority": "medium",
                "total_resolution_value": 0
            }
    
    def _weighted_choice(self, choices: List[tuple]) -> str:
        """Select item from weighted choices list"""
        items, weights = zip(*choices)
        return random.choices(items, weights=weights)[0]
    
    def save_dataset(self, dataset: Dict, filename: str = "customer_service_dataset.json"):
        """Save complete dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
    
    def create_evaluation_format(self, dataset: Dict) -> Dict:
        """Convert dataset to format suitable for asset-based TextEvolve system"""
        
        evaluation_data = {}
        
        for i, ticket in enumerate(dataset["tickets"]):
            # Extract just the customer-facing ticket information (what an agent would receive)
            customer_ticket = {
                "ticket_id": ticket["ticket_id"],
                "date": ticket["date"],
                "sender_email": ticket["customer_info"]["email"],
                "sender_name": ticket["customer_info"].get("name", ""),
                "subject": ticket["subject"],
                "message": ticket["message"]
            }
            
            # Add phone if available
            if "phone" in ticket["customer_info"]:
                customer_ticket["sender_phone"] = ticket["customer_info"]["phone"]
            
            # Add order number if mentioned in the ticket
            if "order_number" in ticket["customer_info"]:
                customer_ticket["order_number"] = ticket["customer_info"]["order_number"]
            
            # The question is just the customer ticket (what the agent receives)
            question = json.dumps(customer_ticket, indent=2)
            
            # The answer is the resolution plan (what the agent should produce)
            answer = json.dumps(ticket["resolution_plan"], indent=2)
            
            evaluation_data[f"support_ticket_{i:03d}"] = {
                "question": question.strip(),
                "answer": answer
            }
        
        return evaluation_data

# Usage Example
if __name__ == "__main__":
    # Initialize generator
    generator = CustomerServiceDataGenerator(seed=42)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(
        num_customers=5,
        num_tickets=5
    )
    
    # Save full dataset
    generator.save_dataset(dataset, "full_customer_service_dataset.json")
    
    # Create evaluation format for ADAS
    eval_data = generator.create_evaluation_format(dataset)
    
    # Save evaluation format
    with open("customer_service_evaluation.json", 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print("Data generation complete!")
    print(f"Generated {len(dataset['customers'])} customers")
    print(f"Generated {len(dataset['orders'])} orders") 
    print(f"Generated {len(dataset['tickets'])} tickets")
    print("Files saved: full_customer_service_dataset.json, customer_service_evaluation.json")
