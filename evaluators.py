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
                "score": 1.0 if match else 0.0,
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
                "score": 1.0 if exact_match else 0.0,
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
            "confidence": f1,  # Also provide as confidence
            "score": 1.0 if match else 0.0,  # Binary by default; change here if continuous desired
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
            "score": 1.0 if match else 0.0,
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
                "score": 1.0 if result.get("match", False) else 0.0,
                "explanation": result.get("explanation", "No explanation provided"),
                "evaluator_type": "ticketworld"
            }
            
        except Exception as e:
            print(f"Error in TicketWorld evaluation: {e}")
            # Simple fallback
            return {
                "match": False,
                "confidence": 0.0,
                "score": 0.0,
                "explanation": f"Evaluation failed due to error: {str(e)}",
                "evaluator_type": "ticketworld"
            }


class HealthBenchEvaluator(Evaluator):
    """
    Evaluator for HealthBench dataset using rubric-based evaluation.
    
    As specified in the HealthBench paper:
    - For each rubric criterion, determine if the response meets it
    - If met: award full points (can be positive or negative)
    - If not met: award 0 points
    - Sum all points and divide by max possible score
    - Final score is clipped to [0, 1]
    """
    
    def __init__(self, llm_caller=None):
        """
        Initialize with LLM caller function.
        If None, will need to be set before use.
        """
        self.llm_caller = llm_caller
    
    def set_llm_caller(self, llm_caller):
        """Set the LLM calling function (injected from AgentSystem)"""
        self.llm_caller = llm_caller
    
    def _evaluate_criterion(self, system_answer: str, criterion: str) -> bool:
        """
        Use LLM to determine if the system answer meets a specific criterion.
        
        Args:
            system_answer: The response to evaluate
            criterion: The criterion description
            
        Returns:
            bool: True if criterion is met, False otherwise
        """
        prompt = f"""You are evaluating a medical AI assistant's response against a specific criterion.

CRITERION TO EVALUATE:
{criterion}

SYSTEM RESPONSE:
{system_answer}

Does the system response meet this criterion? Answer with ONLY "YES" or "NO".

If the criterion describes something the response should do or include, answer YES if the response does it.
If the criterion describes something the response should NOT do (a negative criterion), answer YES if the response violates this rule (meaning points should be deducted).

Answer: """
        
        try:
            response = self.llm_caller(prompt, system_instruction="You are evaluating medical responses against specific criteria. Answer only YES or NO.")
            response = response.strip().upper()
            
            # Extract YES or NO from response
            if "YES" in response:
                return True
            elif "NO" in response:
                return False
            else:
                # Default to False if unclear
                print(f"Warning: Unclear LLM response for criterion evaluation: {response}")
                return False
                
        except Exception as e:
            print(f"Error evaluating criterion: {e}")
            return False
    
    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate using HealthBench rubric-based scoring.
        
        Args:
            system_answer: The system's response
            golden_answer: Not used (HealthBench uses rubrics, not reference answers)
            context: Must contain rubrics in 'meta' dictionary
            
        Returns:
            Dict with evaluation results including per-example score
        """
        
        if not self.llm_caller:
            raise ValueError("LLM caller not set. Call set_llm_caller() first.")
        
        # Extract rubrics from context (they are stored in meta)
        if not context:
            raise ValueError("Context is required for HealthBench evaluation")
        
        meta = context.get('meta', {})
        rubrics = meta.get('rubrics', [])
        
        if not rubrics:
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": "No rubrics provided for evaluation",
                "evaluator_type": "healthbench",
                "score": 0.0,
                "total_points": 0.0,
                "max_possible_points": 0.0,
                "criteria_evaluated": 0
            }
        
        # Calculate max possible score (sum of all positive points)
        max_possible_score = sum(rubric.get('points', 0) for rubric in rubrics if rubric.get('points', 0) > 0)
        
        if max_possible_score == 0:
            # Edge case: no positive points available
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": "No positive points available in rubrics",
                "evaluator_type": "healthbench",
                "score": 0.0,
                "total_points": 0.0,
                "max_possible_points": 0.0,
                "criteria_evaluated": len(rubrics)
            }
        
        # Evaluate each criterion
        total_points = 0.0
        criteria_met = []
        criteria_not_met = []
        
        for rubric in rubrics:
            criterion = rubric.get('criterion', '')
            points = rubric.get('points', 0)
            
            if not criterion:
                continue
            
            # Check if criterion is met
            is_met = self._evaluate_criterion(system_answer, criterion)
            
            if is_met:
                total_points += points
                criteria_met.append({
                    'criterion': criterion,
                    'points': points
                })
            else:
                criteria_not_met.append({
                    'criterion': criterion,
                    'points': points
                })
        
        # Calculate final score: total_points / max_possible_score, clipped to [0, 1]
        raw_score = total_points / max_possible_score
        final_score = max(0.0, min(1.0, raw_score))
        
        # Determine match (using 0.5 as threshold)
        match = final_score >= 0.5
        
        # Create detailed explanation
        explanation = f"HealthBench Score: {final_score:.3f} ({total_points:.1f} / {max_possible_score:.1f} points). "
        explanation += f"Met {len(criteria_met)} criteria, missed {len(criteria_not_met)} criteria."
        
        return {
            "match": match,
            "confidence": final_score,
            "explanation": explanation,
            "evaluator_type": "healthbench",
            "score": final_score,
            "raw_score": raw_score,
            "total_points": total_points,
            "max_possible_points": max_possible_score,
            "criteria_evaluated": len(rubrics),
            "criteria_met": len(criteria_met),
            "criteria_not_met": len(criteria_not_met),
            "criteria_met_details": criteria_met,
            "criteria_not_met_details": criteria_not_met
        }


class HealthBenchOpenAIEvaluator(Evaluator):
    """
    Reference-style HealthBench evaluator modeled after OpenAI's simple-evals implementation.

    Key behavior:
    - Present ALL rubric criteria to the judge in one call
    - The judge returns a JSON array indicating for each criterion whether it's met
      (for negative-point criteria, "met" means the violation occurred and the negative
       points should be applied)
    - Award full points for criteria marked met (including negative points for violations)
    - Final score = sum(awarded_points) / sum(positive_points), clipped to [0, 1]
    - Returns a detailed breakdown compatible with our evaluator interface
    """

    def __init__(self, llm_caller=None):
        self.llm_caller = llm_caller

    def set_llm_caller(self, llm_caller):
        self.llm_caller = llm_caller

    def evaluate(self, system_answer: str, golden_answer: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.llm_caller:
            raise ValueError("LLM caller not set. Call set_llm_caller() first.")

        if not context:
            raise ValueError("Context is required for HealthBench evaluation")

        meta = context.get('meta', {})
        rubrics = meta.get('rubrics', [])

        if not rubrics:
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": "No rubrics provided for evaluation",
                "evaluator_type": "healthbench_openai",
                "score": 0.0,
                "total_points": 0.0,
                "max_possible_points": 0.0,
                "criteria_evaluated": 0
            }

        # Calculate denominator: sum of positive points only
        max_possible_score = sum(r.get('points', 0) for r in rubrics if r.get('points', 0) > 0)
        if max_possible_score == 0:
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": "No positive points available in rubrics",
                "evaluator_type": "healthbench_openai",
                "score": 0.0,
                "total_points": 0.0,
                "max_possible_points": 0.0,
                "criteria_evaluated": len(rubrics)
            }

        # Prepare a compact rubric list for the grader
        rubric_items = []
        for idx, r in enumerate(rubrics):
            rubric_items.append({
                "index": idx,
                "points": r.get('points', 0),
                "criterion": r.get('criterion', '')
            })

        prompt = f"""
You are grading a medical assistant's response against rubric criteria for HealthBench.

SYSTEM RESPONSE:
{system_answer}

RUBRIC CRITERIA:
Provide a decision for each object:
```json
{json.dumps(rubric_items, indent=2)}
```

Instructions:
- For each rubric item, set met=true if the response satisfies the criterion.
- If a criterion is a negative rule (its points value is negative, meaning it describes something the response should NOT do), set met=true only if the response VIOLATES that rule (so the negative points should be applied).
- Return ONLY valid JSON: an array of objects with fields: index (int), met (bool), and optional reason (string).

Output JSON schema example:
[
  {{"index": 0, "met": true, "reason": "Mentions 2-2.4 inches"}},
  {{"index": 1, "met": false}},
  ...
]
"""

        try:
            response = self.llm_caller(prompt, system_instruction="You are a precise grader. Return strictly valid JSON with boolean 'met' decisions.")
            text = response.strip()
            if text.startswith("```json"):
                text = text.split("```json", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            decisions = json.loads(text)
            if not isinstance(decisions, list):
                raise ValueError("Grader output is not a JSON array")

            # Index decisions by rubric index
            idx_to_met = {}
            for item in decisions:
                try:
                    idx = int(item.get("index"))
                    met = bool(item.get("met", False))
                    idx_to_met[idx] = met
                except Exception:
                    continue

            total_points = 0.0
            criteria_met_details = []
            criteria_not_met_details = []

            for i, r in enumerate(rubrics):
                points = r.get('points', 0)
                criterion = r.get('criterion', '')
                met = idx_to_met.get(i, False)
                if met:
                    total_points += points
                    criteria_met_details.append({"criterion": criterion, "points": points})
                else:
                    criteria_not_met_details.append({"criterion": criterion, "points": points})

            raw_score = total_points / max_possible_score
            final_score = max(0.0, min(1.0, raw_score))
            match = final_score >= 0.5

            explanation = (
                f"HealthBench (OpenAI-style) Score: {final_score:.3f} "
                f"({total_points:.1f} / {max_possible_score:.1f} points). "
                f"Met {len(criteria_met_details)} criteria, missed {len(criteria_not_met_details)} criteria."
            )

            return {
                "match": match,
                "confidence": final_score,
                "explanation": explanation,
                "evaluator_type": "healthbench_openai",
                "score": final_score,
                "raw_score": raw_score,
                "total_points": total_points,
                "max_possible_points": max_possible_score,
                "criteria_evaluated": len(rubrics),
                "criteria_met": len(criteria_met_details),
                "criteria_not_met": len(criteria_not_met_details),
                "criteria_met_details": criteria_met_details,
                "criteria_not_met_details": criteria_not_met_details
            }

        except Exception as e:
            print(f"Error in HealthBenchOpenAI evaluation: {e}")
            return {
                "match": False,
                "confidence": 0.0,
                "explanation": f"Evaluation failed due to error: {str(e)}",
                "evaluator_type": "healthbench_openai",
                "score": 0.0,
                "total_points": 0.0,
                "max_possible_points": max_possible_score if 'max_possible_score' in locals() else 0.0,
                "criteria_evaluated": len(rubrics) if 'rubrics' in locals() else 0
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
        "healthbench": HealthBenchEvaluator,
        "healthbench_openai": HealthBenchOpenAIEvaluator,
    }
    
    if evaluator_name not in evaluators:
        print(f"Warning: Unknown evaluator '{evaluator_name}', falling back to 'llm'")
        evaluator_name = "llm"
    
    return evaluators[evaluator_name]()


def list_available_evaluators() -> List[str]:
    """Return list of available evaluator names"""
    return ["llm", "f1", "exact_match", "exact", "ticketworld", "healthbench", "healthbench_openai"] 