#!/usr/bin/env python
"""
evaluators.py - Custom evaluation functions for different dataset types

This module provides a flexible evaluation system that allows different datasets
to use specialized evaluation methods beyond the default LLM-as-judge approach.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set, List
from collections import Counter


class Evaluator(ABC):
    """Base class for all evaluators"""
    
    @abstractmethod
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a system answer against a golden answer.
        
        Args:
            system_answer: The answer produced by the system
            golden_answer: The reference/correct answer
            context: Optional additional context (e.g., question, metadata)
            
        Returns:
            Dict with at minimum: {"match": bool, "confidence": float, "explanation": str}
            Can include additional evaluator-specific metrics
        """
        pass


class LLMEvaluator(Evaluator):
    """LLM-as-judge evaluator (current system default)"""
    
    def __init__(self, llm_caller=None):
        """
        Initialize with LLM caller function.
        If None, will need to be set before use.
        """
        self.llm_caller = llm_caller
    
    def set_llm_caller(self, llm_caller):
        """Set the LLM calling function (injected from AgentSystem)"""
        self.llm_caller = llm_caller
    
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Use LLM to determine if answers are semantically equivalent"""
        
        if not self.llm_caller:
            raise ValueError("LLM caller not set. Call set_llm_caller() first.")
        
        # Role-specific system instruction for the evaluator
        evaluator_system_instruction = "You are now acting as an Answer Evaluator. Your task is to determine if two answers convey the same meaning, even if they are worded or formatted differently."

        prompt = f"""
        You're evaluating two answers to determine if they convey the same information.

        System answer: {system_answer}
        Golden answer: {golden_answer}

        Do these answers communicate the same information, even if worded or formatted differently? The "Golden answer" is the reference answer. The "System answer" produced by our system may contain reasoning traces, but your job is to verify if in the system answer there is an answer that is semantically equivalent to the golden answer.
        If this is a detailed numerical answer, where clearly precision is required, check very close, element by element, to ensure that the content is correct.
        A system answer that is just code, or code that is meant to produce the final answer but with no final output is not acceptable. The system answer must contain a final answer that is semantically equivalent to the golden answer.
        Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "reason"}}
        """
        
        try:
            response = self.llm_caller(prompt, system_instruction=evaluator_system_instruction)

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]

            result = json.loads(response)

            # Extract match information
            match = result.get("match", False)
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "No explanation provided")

            return {
                "match": match,
                "confidence": confidence,
                "explanation": explanation,
                "evaluator_type": "llm"
            }
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            # Fallback to exact match
            exact_match = system_answer.strip() == golden_answer.strip()
            return {
                "match": exact_match,
                "confidence": 1.0 if exact_match else 0.0,
                "explanation": f"Fallback to exact match comparison due to LLM error: {str(e)}",
                "evaluator_type": "llm"
            }


class F1ScoreEvaluator(Evaluator):
    """F1 score-based evaluator for text similarity"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize F1 evaluator.
        
        Args:
            threshold: F1 score threshold for considering a match (0.5 default)
        """
        self.threshold = threshold
    
    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization - split on whitespace and punctuation, lowercase"""
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = set(cleaned.split())
        return tokens
    
    def _calculate_f1(self, system_tokens: Set[str], golden_tokens: Set[str]) -> tuple:
        """Calculate precision, recall, and F1 score"""
        if not golden_tokens:
            return 0.0, 0.0, 0.0
        
        if not system_tokens:
            return 0.0, 0.0, 0.0
        
        # Calculate intersection
        intersection = system_tokens.intersection(golden_tokens)
        
        # Calculate precision and recall
        precision = len(intersection) / len(system_tokens) if system_tokens else 0.0
        recall = len(intersection) / len(golden_tokens) if golden_tokens else 0.0
        
        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate using F1 score between tokenized answers"""
        
        # Tokenize both answers
        system_tokens = self._tokenize(system_answer)
        golden_tokens = self._tokenize(golden_answer)
        
        # Calculate metrics
        precision, recall, f1 = self._calculate_f1(system_tokens, golden_tokens)
        
        # Determine match based on threshold
        match = f1 >= self.threshold
        
        # Create explanation
        explanation = f"F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f} (threshold={self.threshold})"
        
        return {
            "match": match,
            "confidence": f1,  # Use F1 score as confidence
            "explanation": explanation,
            "evaluator_type": "f1",
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "threshold": self.threshold
        }


class ExactMatchEvaluator(Evaluator):
    """Exact string match evaluator"""
    
    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        """
        Initialize exact match evaluator.
        
        Args:
            case_sensitive: Whether to do case-sensitive matching
            strip_whitespace: Whether to strip leading/trailing whitespace
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
    
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate using exact string matching"""
        
        # Prepare answers for comparison
        sys_ans = system_answer
        gold_ans = golden_answer
        
        if self.strip_whitespace:
            sys_ans = sys_ans.strip()
            gold_ans = gold_ans.strip()
        
        if not self.case_sensitive:
            sys_ans = sys_ans.lower()
            gold_ans = gold_ans.lower()
        
        # Check match
        match = sys_ans == gold_ans
        
        # Create explanation
        explanation = f"Exact match ({'case-sensitive' if self.case_sensitive else 'case-insensitive'}, {'whitespace-sensitive' if not self.strip_whitespace else 'whitespace-stripped'})"
        
        return {
            "match": match,
            "confidence": 1.0 if match else 0.0,
            "explanation": explanation,
            "evaluator_type": "exact_match",
            "case_sensitive": self.case_sensitive,
            "strip_whitespace": self.strip_whitespace
        }


class TicketWorldEvaluator(Evaluator):
    """Simplified evaluator for TicketWorld customer service resolutions"""
    
    def __init__(self, llm_caller=None):
        """
        Initialize with LLM caller function.
        If None, will need to be set before use.
        """
        self.llm_caller = llm_caller
    
    def set_llm_caller(self, llm_caller):
        """Set the LLM calling function (injected from AgentSystem)"""
        self.llm_caller = llm_caller
    
    def _extract_key_fields_from_golden(self, golden_text: str) -> str:
        """Extract key fields from golden answer for comparison"""
        import json
        import re
        
        try:
            # Try to parse as JSON first
            json_match = re.search(r'\{.*\}', golden_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    key_info = []
                    
                    # Order ID
                    if data.get("order_id"):
                        key_info.append(f"Order ID: {data['order_id']}")
                    
                    # Customer ID 
                    customer_id = data.get("customer_lookup", {}).get("customer_id") or data.get("customer_id")
                    if customer_id:
                        key_info.append(f"Customer ID: {customer_id}")
                    
                    # Actions
                    actions = data.get("actions", [])
                    if actions:
                        action_types = [action.get("type") for action in actions if isinstance(action, dict) and action.get("type")]
                        if action_types:
                            key_info.append(f"Actions: {', '.join(action_types)}")
                    
                    # Escalation
                    if data.get("escalation_required") is not None:
                        key_info.append(f"Escalation Required: {data['escalation_required']}")
                    
                    # Policies
                    policies = data.get("policy_references", [])
                    if policies:
                        key_info.append(f"Policy References: {', '.join(policies)}")
                    
                    return "\n".join(key_info)
        except:
            pass
        
        # Fallback: return original text
        return golden_text
    
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Simple evaluation: does the system output contain the right aspects of the golden answer?
        """
        
        if not self.llm_caller:
            raise ValueError("LLM caller not set. Call set_llm_caller() first.")
        
        # Extract key fields from golden answer
        key_fields = self._extract_key_fields_from_golden(golden_answer)
        
        # Create simple evaluation prompt
        prompt = f"""
You are evaluating a customer service resolution. The golden answer contains the correct resolution fields, and you need to check if the system output contains the right aspects.

KEY FIELDS FROM CORRECT RESOLUTION:
{key_fields}

SYSTEM OUTPUT:
{system_answer}

Does the system output contain the right aspects of the correct resolution? Consider:- Are the correct IDs (order, customer) mentioned? If these are provided for the golden answer (and not N/A) then the system output MUST match.
- Are appropriate actions taken?
- Is the escalation decision reasonable?
- Are relevant policies referenced?

Exact phrasing or formatting doesn't matter - focus on whether the system captured the essential elements of the correct resolution.

Return only a JSON object with: {{"match": true/false, "confidence": 0-1, "explanation": "brief reason"}}
"""
        
        try:
            response = self.llm_caller(prompt, system_instruction="You are evaluating whether a customer service response contains the right key elements. Focus on substance, not formatting.")
            
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.endswith("```"):
                response = response.split("```")[0]
            
            result = json.loads(response)
            
            return {
                "match": result.get("match", False),
                "confidence": result.get("confidence", 0.0),
                "explanation": result.get("explanation", "No explanation provided"),
                "evaluator_type": "ticketworld"
            }
            
        except Exception as e:
            print(f"Error in TicketWorld evaluation: {e}")
            # Simple fallback
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": f"Evaluation failed due to error: {str(e)}",
                "evaluator_type": "ticketworld"
            }


def create_evaluator(evaluator_name: str) -> Evaluator:
    """
    Factory function to create evaluator instances.
    
    Args:
        evaluator_name: Name of the evaluator to create
        
    Returns:
        Evaluator instance
        
    Raises:
        ValueError: If evaluator name is not recognized
    """
    evaluators = {
        "llm": LLMEvaluator,
        "f1": F1ScoreEvaluator,
        "exact_match": ExactMatchEvaluator,
        "exact": ExactMatchEvaluator,  # Alias
        "ticketworld": TicketWorldEvaluator,
    }
    
    if evaluator_name not in evaluators:
        print(f"Warning: Unknown evaluator '{evaluator_name}', falling back to 'llm'")
        evaluator_name = "llm"
    
    return evaluators[evaluator_name]()


def list_available_evaluators() -> List[str]:
    """Return list of available evaluator names"""
    return ["llm", "f1", "exact_match", "exact", "ticketworld"] 