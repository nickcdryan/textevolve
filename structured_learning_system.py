import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class StructuredLearningSystem:
    """
    Manages structured learnings that can be queried and used for conditional decision-making.
    Complements the existing learnings.txt with structured, programmatically accessible insights.
    """
    
    def __init__(self, learning_file: str = "structured_learnings.json"):
        self.learning_file = Path(learning_file)
        self.learnings = self._load_learnings()
    
    def _load_learnings(self) -> List[Dict]:
        """Load existing structured learnings from file."""
        if self.learning_file.exists():
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("learnings", [])
            except Exception as e:
                print(f"Error loading structured learnings: {e}")
                return []
        return []
    
    def _save_learnings(self):
        """Save structured learnings to file."""
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "learnings": self.learnings
            }
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving structured learnings: {e}")
    
    def add_learning(self, 
                    iteration: int,
                    script_analysis: Dict,
                    complexity_assessment: Dict,
                    error_analysis: Dict,
                    performance_data: Dict,
                    dataset_context: Dict = None) -> str:
        """
        Add a new structured learning entry based on iteration results.
        
        Args:
            iteration: Iteration number
            script_analysis: Deep script analysis results
            complexity_assessment: Complexity assessment results
            error_analysis: Error analysis results
            performance_data: Accuracy, success rate, etc.
            dataset_context: Information about the dataset/task type
            
        Returns:
            learning_id: Unique identifier for this learning
        """
        learning_id = f"learning_{iteration}_{int(datetime.now().timestamp())}"
        
        # Extract key conditions and context
        conditions = {
            "task_complexity": complexity_assessment.get("task_complexity", "MODERATE"),
            "approach_complexity": complexity_assessment.get("approach_complexity", "MODERATE"),
            "architectural_pattern": script_analysis.get("architectural_pattern", "unknown"),
            "implementation_quality": script_analysis.get("implementation_quality", "FAIR"),
            "tool_usage_efficiency": script_analysis.get("tool_usage_efficiency", "GOOD"),
            "llm_integration_pattern": script_analysis.get("llm_integration_pattern", "unknown")
        }
        
        if dataset_context:
            conditions.update({
                "dataset_type": dataset_context.get("type", "unknown"),
                "has_database": dataset_context.get("has_database", False),
                "requires_reasoning": dataset_context.get("requires_reasoning", True)
            })
        
        # Determine learning type and lesson
        learning_type, lesson = self._extract_lesson(
            script_analysis, complexity_assessment, error_analysis, performance_data)
        
        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(performance_data, script_analysis)
        
        # Determine transferability
        transferability = script_analysis.get("transferability", "MEDIUM")
        
        learning_entry = {
            "learning_id": learning_id,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "learning_type": learning_type,
            "lesson": lesson,
            "conditions": conditions,
            "evidence_strength": evidence_strength,
            "transferability": transferability,
            "performance_data": {
                "accuracy": performance_data.get("accuracy", 0.0),
                "success_rate": performance_data.get("success_rate", 0.0),
                "error_count": performance_data.get("error_count", 0)
            },
            "failure_modes": script_analysis.get("failure_modes", []),
            "improvement_opportunities": script_analysis.get("improvement_opportunities", []),
            "tags": self._generate_tags(script_analysis, complexity_assessment, error_analysis)
        }
        
        self.learnings.append(learning_entry)
        self._save_learnings()
        
        print(f"Added structured learning: {learning_type} - {lesson[:100]}...")
        return learning_id
    
    def _extract_lesson(self, script_analysis: Dict, complexity_assessment: Dict, 
                       error_analysis: Dict, performance_data: Dict) -> tuple:
        """Extract the key lesson and categorize learning type."""
        
        accuracy = performance_data.get("accuracy", 0.0)
        complexity_match = complexity_assessment.get("match_assessment", "APPROPRIATE")
        implementation_quality = script_analysis.get("implementation_quality", "FAIR")
        
        # Determine learning type and lesson based on patterns
        if accuracy < 0.1 and "runtime" in str(error_analysis.get("runtime_errors", [])).lower():
            learning_type = "IMPLEMENTATION_BUG"
            lesson = f"Script with {script_analysis.get('architectural_pattern', 'unknown')} pattern failed due to implementation issues, not conceptual problems"
            
        elif complexity_match == "OVER_ENGINEERED" and accuracy < 0.5:
            learning_type = "COMPLEXITY_MISMATCH"  
            lesson = f"Over-engineered {script_analysis.get('architectural_pattern', 'unknown')} approach ineffective for {complexity_assessment.get('task_complexity', 'MODERATE')} complexity tasks"
            
        elif complexity_match == "UNDER_ENGINEERED" and accuracy < 0.5:
            learning_type = "COMPLEXITY_MISMATCH"
            lesson = f"Under-engineered {script_analysis.get('architectural_pattern', 'unknown')} approach insufficient for {complexity_assessment.get('task_complexity', 'MODERATE')} complexity tasks"
            
        elif accuracy > 0.7 and implementation_quality in ["EXCELLENT", "GOOD"]:
            learning_type = "SUCCESSFUL_PATTERN"
            lesson = f"{script_analysis.get('architectural_pattern', 'unknown')} pattern with {script_analysis.get('llm_integration_pattern', 'unknown')} LLM integration effective for this task type"
            
        elif accuracy > 0.3 and accuracy < 0.7:
            learning_type = "PARTIAL_SUCCESS"
            lesson = f"{script_analysis.get('architectural_pattern', 'unknown')} approach shows promise but needs refinement in {', '.join(script_analysis.get('improvement_opportunities', [])[:2])}"
            
        else:
            learning_type = "GENERAL_OBSERVATION"
            lesson = f"Script using {script_analysis.get('architectural_pattern', 'unknown')} pattern achieved {accuracy:.2f} accuracy with {implementation_quality} implementation quality"
            
        return learning_type, lesson
    
    def _calculate_evidence_strength(self, performance_data: Dict, script_analysis: Dict) -> str:
        """Calculate how strong the evidence is for this learning."""
        accuracy = performance_data.get("accuracy", 0.0)
        success_rate = performance_data.get("success_rate", 0.0)
        implementation_quality = script_analysis.get("implementation_quality", "FAIR")
        
        # Strong evidence: high/low performance with good implementation
        if (accuracy > 0.8 or accuracy < 0.1) and implementation_quality in ["EXCELLENT", "GOOD"]:
            return "STRONG"
        # Medium evidence: clear patterns but moderate performance  
        elif accuracy > 0.4 or success_rate > 0.7:
            return "MEDIUM"
        else:
            return "WEAK"
    
    def _generate_tags(self, script_analysis: Dict, complexity_assessment: Dict, error_analysis: Dict) -> List[str]:
        """Generate searchable tags for this learning."""
        tags = []
        
        # Pattern-based tags
        if script_analysis.get("architectural_pattern") != "unknown":
            tags.append(f"pattern:{script_analysis['architectural_pattern']}")
        
        if script_analysis.get("llm_integration_pattern") != "unknown":  
            tags.append(f"llm:{script_analysis['llm_integration_pattern']}")
            
        # Complexity tags
        tags.append(f"task_complexity:{complexity_assessment.get('task_complexity', 'MODERATE')}")
        tags.append(f"complexity_match:{complexity_assessment.get('match_assessment', 'APPROPRIATE')}")
        
        # Quality tags
        tags.append(f"quality:{script_analysis.get('implementation_quality', 'FAIR')}")
        tags.append(f"tool_usage:{script_analysis.get('tool_usage_efficiency', 'GOOD')}")
        
        # Error-based tags
        if error_analysis.get("runtime_errors"):
            tags.append("has_runtime_errors")
        if error_analysis.get("primary_issue"):
            tags.append("has_primary_issue")
            
        return tags
    
    def query_learnings(self, 
                       conditions: Dict = None, 
                       learning_types: List[str] = None,
                       tags: List[str] = None,
                       min_evidence_strength: str = None,
                       limit: int = 10) -> List[Dict]:
        """
        Query structured learnings based on conditions and filters.
        
        Args:
            conditions: Dict of condition key-value pairs to match
            learning_types: List of learning types to filter by
            tags: List of tags that must be present
            min_evidence_strength: Minimum evidence strength (WEAK/MEDIUM/STRONG)
            limit: Maximum number of results
            
        Returns:
            List of matching learning entries
        """
        results = []
        evidence_order = {"WEAK": 0, "MEDIUM": 1, "STRONG": 2}
        min_strength_level = evidence_order.get(min_evidence_strength, -1)
        
        for learning in self.learnings:
            # Check learning type filter
            if learning_types and learning.get("learning_type") not in learning_types:
                continue
                
            # Check evidence strength filter
            if min_evidence_strength:
                learning_strength = evidence_order.get(learning.get("evidence_strength", "WEAK"), 0)
                if learning_strength < min_strength_level:
                    continue
            
            # Check conditions filter
            if conditions:
                learning_conditions = learning.get("conditions", {})
                if not all(learning_conditions.get(k) == v for k, v in conditions.items()):
                    continue
            
            # Check tags filter
            if tags:
                learning_tags = learning.get("tags", [])
                if not all(tag in learning_tags for tag in tags):
                    continue
            
            results.append(learning)
        
        # Sort by evidence strength and recency
        results.sort(key=lambda x: (
            evidence_order.get(x.get("evidence_strength", "WEAK"), 0),
            x.get("iteration", 0)
        ), reverse=True)
        
        return results[:limit]
    
    def get_conditional_guidance(self, current_conditions: Dict) -> Dict:
        """
        Get guidance based on current conditions by finding similar past learnings.
        
        Args:
            current_conditions: Current script/task conditions
            
        Returns:
            Dict with recommendations and supporting evidence
        """
        # Find relevant learnings
        relevant_learnings = self.query_learnings(
            conditions=current_conditions,
            min_evidence_strength="MEDIUM",
            limit=5
        )
        
        if not relevant_learnings:
            # Broaden search by removing some conditions
            partial_conditions = {k: v for k, v in current_conditions.items() 
                                if k in ["task_complexity", "architectural_pattern"]}
            relevant_learnings = self.query_learnings(
                conditions=partial_conditions,
                min_evidence_strength="WEAK", 
                limit=3
            )
        
        # Aggregate insights
        recommendations = []
        supporting_evidence = []
        
        for learning in relevant_learnings:
            if learning.get("learning_type") == "SUCCESSFUL_PATTERN":
                recommendations.append(f"Consider using {learning['conditions'].get('architectural_pattern', 'unknown')} pattern - shown to work well for similar conditions")
            elif learning.get("learning_type") == "COMPLEXITY_MISMATCH":
                recommendations.append(f"Avoid complexity mismatch - {learning['lesson']}")
            elif learning.get("learning_type") == "IMPLEMENTATION_BUG":
                recommendations.append(f"Focus on implementation quality - similar patterns failed due to bugs")
            
            supporting_evidence.append({
                "iteration": learning.get("iteration"),
                "lesson": learning.get("lesson"),
                "evidence_strength": learning.get("evidence_strength"),
                "accuracy": learning.get("performance_data", {}).get("accuracy", 0)
            })
        
        return {
            "recommendations": recommendations[:3],
            "supporting_evidence": supporting_evidence,
            "confidence": "HIGH" if len(relevant_learnings) >= 2 else "MEDIUM" if relevant_learnings else "LOW"
        }
    
    def get_learning_summary(self) -> Dict:
        """Get a summary of all learnings for reporting."""
        if not self.learnings:
            return {"total_learnings": 0}
            
        learning_types = {}
        evidence_strengths = {}
        recent_learnings = []
        
        for learning in self.learnings:
            # Count learning types
            ltype = learning.get("learning_type", "UNKNOWN")
            learning_types[ltype] = learning_types.get(ltype, 0) + 1
            
            # Count evidence strengths
            strength = learning.get("evidence_strength", "WEAK")
            evidence_strengths[strength] = evidence_strengths.get(strength, 0) + 1
            
            # Collect recent learnings
            if learning.get("iteration", 0) >= max(l.get("iteration", 0) for l in self.learnings) - 3:
                recent_learnings.append({
                    "iteration": learning.get("iteration"),
                    "type": learning.get("learning_type"),
                    "lesson": learning.get("lesson", "")[:100] + "..."
                })
        
        return {
            "total_learnings": len(self.learnings),
            "learning_types": learning_types,
            "evidence_strengths": evidence_strengths,
            "recent_learnings": recent_learnings[-5:],  # Last 5 learnings
            "high_confidence_learnings": len([l for l in self.learnings if l.get("evidence_strength") == "STRONG"])
        }