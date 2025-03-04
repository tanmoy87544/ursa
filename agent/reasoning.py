"""
Reasoning components for the scientific agent framework.
Handles problem formalization, component identification, and summarization.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Reasoner:
    """Reasoning capabilities for scientific problem-solving."""
    
    def __init__(self, llm):
        """
        Initialize the reasoner.
        
        Args:
            llm: LLM provider to use for reasoning tasks
        """
        self.llm = llm
    
    def formalize_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Transform an unstructured problem into a formal representation.
        
        Args:
            problem_statement: Natural language description of the problem
            
        Returns:
            Structured representation of the problem
        """
        logger.info("Formalizing problem")
        
        # Construct prompt for problem formalization
        prompt = f"""
        You are a scientific problem-solving assistant with expertise in scientific methodology.
        
        Please analyze the following scientific problem statement and formalize it into a structured representation:
        
        PROBLEM STATEMENT:
        {problem_statement}
        
        Please provide a structured representation of this problem, including:
        1. A clear, concise restatement of the problem
        2. The overall goal/objective to achieve
        3. Primary research questions to answer
        4. Hypotheses to test, if applicable
        5. Success criteria for a solution
        
        Format your response as a JSON with the following schema:
        {{
            "problem_restatement": "Clear restatement of the problem",
            "objective": "Overall goal",
            "research_questions": ["Question 1", "Question 2", ...],
            "hypotheses": ["Hypothesis 1", "Hypothesis 2", ...],
            "success_criteria": ["Criterion 1", "Criterion 2", ...]
        }}
        """
        
        try:
            formalized = self.llm.generate_with_json_output(prompt, {
                "type": "object",
                "properties": {
                    "problem_restatement": {"type": "string"},
                    "objective": {"type": "string"},
                    "research_questions": {"type": "array", "items": {"type": "string"}},
                    "hypotheses": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}}
                }
            })
            
            logger.debug(f"Formalized problem: {json.dumps(formalized, indent=2)}")
            return formalized
            
        except Exception as e:
            logger.error(f"Error in problem formalization: {str(e)}")
            # Return a minimal formalized representation
            return {
                "problem_restatement": problem_statement,
                "objective": "Solve the given problem",
                "research_questions": [],
                "hypotheses": [],
                "success_criteria": ["Provide a solution to the problem"]
            }
    
    def identify_components(self, formalized_problem: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Identify key components of the problem.
        
        Args:
            formalized_problem: Structured representation of the problem
            
        Returns:
            Dictionary containing variables, constraints, domains, and approaches
        """
        logger.info("Identifying problem components")
        
        # Construct prompt for component identification
        prompt = f"""
        Based on the following formalized scientific problem:
        {json.dumps(formalized_problem, indent=2)}
        
        Please identify and list the following key components:
        
        1. VARIABLES: Key variables that need to be measured, controlled, or optimized
        2. CONSTRAINTS: Limitations or constraints that must be respected
        3. DOMAINS: Scientific domains and sub-fields relevant to this problem
        4. APPROACHES: Potential scientific approaches or methodologies that could be applied
        
        Format your response as a JSON with the following schema:
        {{
            "variables": ["Variable 1", "Variable 2", ...],
            "constraints": ["Constraint 1", "Constraint 2", ...],
            "domains": ["Domain 1", "Domain 2", ...],
            "approaches": ["Approach 1", "Approach 2", ...]
        }}
        """
        
        try:
            components = self.llm.generate_with_json_output(prompt, {
                "type": "object",
                "properties": {
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "domains": {"type": "array", "items": {"type": "string"}},
                    "approaches": {"type": "array", "items": {"type": "string"}}
                }
            })
            
            logger.debug(f"Identified components: {json.dumps(components, indent=2)}")
            return components
            
        except Exception as e:
            logger.error(f"Error in component identification: {str(e)}")
            # Return empty components
            return {
                "variables": [],
                "constraints": [],
                "domains": [],
                "approaches": []
            }
    
    def generate_hypothesis(self, formalized_problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate testable hypotheses about the problem.
        
        Args:
            formalized_problem: Structured representation of the problem
            
        Returns:
            List of hypothesis objects with predictions
        """
        logger.info("Generating hypotheses")
        
        if "hypotheses" in formalized_problem and formalized_problem["hypotheses"]:
            # Already has hypotheses in the formalized problem
            raw_hypotheses = formalized_problem["hypotheses"]
            
            # Generate detailed versions with predictions
            structured_hypotheses = []
            for i, hyp in enumerate(raw_hypotheses):
                hypothesis_prompt = f"""
                For the following scientific hypothesis:
                "{hyp}"
                
                Please elaborate on this hypothesis by:
                1. Reformulating it as a clear if-then statement
                2. Listing specific, testable predictions that follow from it
                3. Suggesting how it could be tested
                
                Format your response as a JSON object with the following structure:
                {{
                    "hypothesis_statement": "Clear if-then statement",
                    "predictions": ["Prediction 1", "Prediction 2", ...],
                    "testing_approach": "Brief description of how to test"
                }}
                """
                
                try:
                    result = self.llm.generate_with_json_output(hypothesis_prompt, {
                        "type": "object",
                        "properties": {
                            "hypothesis_statement": {"type": "string"},
                            "predictions": {"type": "array", "items": {"type": "string"}},
                            "testing_approach": {"type": "string"}
                        }
                    })
                    structured_hypotheses.append(result)
                except Exception as e:
                    logger.error(f"Error elaborating hypothesis {i+1}: {str(e)}")
                    structured_hypotheses.append({
                        "hypothesis_statement": hyp,
                        "predictions": [],
                        "testing_approach": "Not specified"
                    })
            
            return structured_hypotheses
            
        else:
            # No hypotheses provided, generate from scratch
            hypothesis_prompt = f"""
            Based on the following scientific problem:
            {json.dumps(formalized_problem, indent=2)}
            
            Please generate 2-3 testable scientific hypotheses that could explain or address this problem.
            For each hypothesis:
            1. Formulate it as a clear if-then statement
            2. List specific, testable predictions that follow from it
            3. Suggest how it could be tested
            
            Format your response as a JSON array with objects having the following structure:
            [
                {{
                    "hypothesis_statement": "Clear if-then statement for hypothesis 1",
                    "predictions": ["Prediction 1", "Prediction 2", ...],
                    "testing_approach": "Brief description of how to test"
                }},
                {{
                    "hypothesis_statement": "Clear if-then statement for hypothesis 2",
                    "predictions": ["Prediction 1", "Prediction 2", ...],
                    "testing_approach": "Brief description of how to test"
                }}
            ]
            """
            
            try:
                hypotheses = self.llm.generate_with_json_output(hypothesis_prompt, {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_statement": {"type": "string"},
                            "predictions": {"type": "array", "items": {"type": "string"}},
                            "testing_approach": {"type": "string"}
                        }
                    }
                })
                return hypotheses
            except Exception as e:
                logger.error(f"Error generating hypotheses: {str(e)}")
                # Return a default hypothesis
                return [{
                    "hypothesis_statement": "If the problem is addressed using standard methods, then a solution can be found.",
                    "predictions": ["A solution will be found using standard approaches"],
                    "testing_approach": "Apply standard methods and evaluate results"
                }]
    
    def analyze_results(self, hypothesis: Dict[str, Any], experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze experimental results in context of hypotheses.
        
        Args:
            hypothesis: The hypothesis being tested
            experimental_results: Results from experiments
            
        Returns:
            Analysis of results with conclusions
        """
        logger.info("Analyzing experimental results")
        
        analysis_prompt = f"""
        You are analyzing the results of a scientific experiment.
        
        HYPOTHESIS TESTED:
        {json.dumps(hypothesis, indent=2)}
        
        EXPERIMENTAL RESULTS:
        {json.dumps(experimental_results, indent=2)}
        
        Please analyze these results in relation to the hypothesis by:
        1. Determining whether each prediction was supported or not
        2. Evaluating the overall support for the hypothesis
        3. Identifying any unexpected findings
        4. Suggesting possible explanations for the results
        5. Recommending next steps
        
        Format your response as a JSON with the following schema:
        {{
            "predictions_evaluation": [
                {{"prediction": "Prediction text", "supported": true/false, "explanation": "Why supported or not"}}
            ],
            "overall_hypothesis_support": "Strong/Moderate/Weak/None",
            "confidence": 0-100,
            "unexpected_findings": ["Finding 1", "Finding 2", ...],
            "possible_explanations": ["Explanation 1", "Explanation 2", ...],
            "next_steps": ["Step 1", "Step 2", ...]
        }}
        """
        
        try:
            analysis = self.llm.generate_with_json_output(analysis_prompt, {
                "type": "object",
                "properties": {
                    "predictions_evaluation": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prediction": {"type": "string"},
                                "supported": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            }
                        }
                    },
                    "overall_hypothesis_support": {"type": "string"},
                    "confidence": {"type": "number"},
                    "unexpected_findings": {"type": "array", "items": {"type": "string"}},
                    "possible_explanations": {"type": "array", "items": {"type": "string"}},
                    "next_steps": {"type": "array", "items": {"type": "string"}}
                }
            })
            
            logger.debug(f"Results analysis: {json.dumps(analysis, indent=2)}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in results analysis: {str(e)}")
            # Return minimal analysis
            return {
                "predictions_evaluation": [],
                "overall_hypothesis_support": "Unknown",
                "confidence": 0,
                "unexpected_findings": [],
                "possible_explanations": ["Analysis could not be completed"],
                "next_steps": ["Review raw data and retry analysis"]
            }
    
    def assess_completeness(self, 
                        problem_statement: str,
                        formalized_problem: Dict[str, Any],
                        solution_plan: Dict[str, Any],
                        execution_results: Dict[str, Any],
                        execution_status: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the scientific investigation.
        
        Args:
            problem_statement: Original problem statement
            formalized_problem: Structured problem representation
            solution_plan: Solution plan with steps
            execution_results: Results from executed steps
            execution_status: Status of execution (completed, failed, etc.)
            
        Returns:
            Summary assessment of whether the goals have been completed
        """
        logger.info("Assessing whether goals have been met")
        
        summary_prompt = f"""
        You are a scientific critic assessing whether the current results met the goals that were set out.
        
        ORIGINAL PROBLEM:
        {problem_statement}
        
        FORMALIZED PROBLEM:
        {json.dumps(formalized_problem, indent=2)}
        
        SOLUTION APPROACH:
        {json.dumps(solution_plan, indent=2)}
        
        EXECUTION RESULTS:
        {json.dumps(execution_results, indent=2)}
        
        EXECUTION STATUS:
        {execution_status}
        
        Thnk through the solution approach and execution results and decide whether the results meet the problem goals.
        
        Format your response as a JSON with the following schema:\
        {{
            "Objective Met?": True or False,
            "Reasoning": "Why you have decided the objective was or wasnt met"
        }}
        """
        
        try:
            summary = self.llm.generate_with_json_output(summary_prompt, {
                "type": "object",
                "properties": {
                    "Objective Met?": {"type": "bool"},
                    "Reasoning": {"type": "string"}
                }
            })
            
            logger.debug(f"Decided if we are done: {json.dumps(summary, indent=2)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Return minimal summary
            return {"properties": {
                    "Objective Met?": {"type": "bool"},
                    "Reasoning": {"type": "string"}}}

    def generate_summary(self, 
                        problem_statement: str,
                        formalized_problem: Dict[str, Any],
                        solution_plan: Dict[str, Any],
                        execution_results: Dict[str, Any],
                        execution_status: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the scientific investigation.
        
        Args:
            problem_statement: Original problem statement
            formalized_problem: Structured problem representation
            solution_plan: Solution plan with steps
            execution_results: Results from executed steps
            execution_status: Status of execution (completed, failed, etc.)
            
        Returns:
            Summary dictionary with key findings, impact, and limitations
        """
        logger.info("Generating solution summary")
        
        summary_prompt = f"""
        You are a scientific expert summarizing the results of a scientific investigation.
        
        ORIGINAL PROBLEM:
        {problem_statement}
        
        FORMALIZED PROBLEM:
        {json.dumps(formalized_problem, indent=2)}
        
        SOLUTION APPROACH:
        {json.dumps(solution_plan, indent=2)}
        
        EXECUTION RESULTS:
        {json.dumps(execution_results, indent=2)}
        
        EXECUTION STATUS:
        {execution_status}
        
        Please create a comprehensive yet concise summary of this scientific investigation.
        Cover the following aspects:
        
        1. Problem overview and significance
        2. Approach taken and methodology
        3. Key findings and results
        4. How well the findings address the original problem
        5. Limitations of the approach and results
        6. Potential impact of the findings
        7. Recommendations for future work
        
        Format your response as a JSON with the following schema:
        {{
            "problem_overview": "Brief restatement of problem and its significance",
            "approach_summary": "Summary of the approach and methodology",
            "key_findings": ["Finding 1", "Finding 2", ...],
            "problem_resolution": "How findings address the original problem",
            "limitations": ["Limitation 1", "Limitation 2", ...],
            "potential_impact": ["Impact 1", "Impact 2", ...],
            "future_work": ["Recommendation 1", "Recommendation 2", ...]
        }}
        """
        
        try:
            summary = self.llm.generate_with_json_output(summary_prompt, {
                "type": "object",
                "properties": {
                    "problem_overview": {"type": "string"},
                    "approach_summary": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "problem_resolution": {"type": "string"},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                    "potential_impact": {"type": "array", "items": {"type": "string"}},
                    "future_work": {"type": "array", "items": {"type": "string"}}
                }
            })
            
            logger.debug(f"Generated summary: {json.dumps(summary, indent=2)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Return minimal summary
            return {
                "problem_overview": problem_statement,
                "approach_summary": "Analysis performed",
                "key_findings": ["Results obtained but summary generation failed"],
                "problem_resolution": "Unknown",
                "limitations": ["Summary generation failed, see raw results"],
                "potential_impact": [],
                "future_work": ["Review raw results and retry summary generation"]
            }
