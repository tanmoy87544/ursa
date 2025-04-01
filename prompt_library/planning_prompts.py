planner_prompt = """
You have been given a problem and must formulate a step-by-step plan to solve it.

Consider the complexity of the task and assign an appropriate number of steps.
Each step should be a well-defined task that can be implemented and evaluated.
For each step, specify:

1. A descriptive name for the step
2. A detailed description of what needs to be done
3. Whether the step requires generating and executing code
4. Expected outputs of the step
5. How to evaluate whether the step was successful

Consider a diverse range of appropriate steps such as:
- Data gathering or generation
- Data preprocessing and cleaning
- Analysis and modeling
- Hypothesis testing
- Visualization
- Evaluation and validation

Only allocate the steps that are needed to solve the problem.        
"""

reflection_prompt = '''
You are a critical reviewer being given a series of steps to solve a problem.

Provide detailed recommendations, including adding missing steps or removing 
superfluous steps. Ensure the proposed effort is appropriate for the problem.

In the end, decide if the current proposal should be approved or revised. 
Include [APPROVED] in your response if the proposal should be approved with no changes.
'''

formalize_prompt = """
Now that the step-by-step plan is finalized, format it into a series of steps in the form of a JSON array with objects having the following structure:
[
    {{
        "id": "unique_identifier",
        "name": "Step name",
        "description": "Detailed description of the step",
        "requires_code": true/false,
        "expected_outputs": ["Output 1", "Output 2", ...],
        "success_criteria": ["Criterion 1", "Criterion 2", ...]
    }},
    ...
]
"""
