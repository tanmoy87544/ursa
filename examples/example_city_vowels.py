import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import ExecutionAgent, PlanningAgent


def main(mode: str):
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        model = ChatLiteLLM(
            model="openai/o3-mini"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=10000 if mode == "prod" else 4000,
            max_retries=2,
        )
        init = {"messages": [HumanMessage(content=problem)]}

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

        # Solve the problem
        planning_output = planner.action.invoke(init)
        print(planning_output["messages"][-1].content)
        planning_output["workspace"] = "workspace_cityVowels"
        final_results = executor.action.invoke(planning_output)
        for x in final_results["messages"]:
            print(x.content)
        # print(final_results["messages"][-1].content)

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    mode = "dev" if sys.argv[-1] == "dev" else "prod"
    final_output = main(mode=mode)  # dev or prod
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print(final_output)

# Solving problem: Find a city with as least 10 vowels in its name.

# [
#     {
#         "id": "step-1",
#         "name": "Interpret Requirements",
#         "description": "Read and clarify the problem condition 'at least 10 vowels in a city name'. Decide whether vowels include only A, E, I, O, U (in any case) or if accented characters count as well.",
#         "requires_code": false,
#         "expected_outputs": ["Clear definition of vowel set (e.g., A, E, I, O, U)"],
#         "success_criteria": ["Criteria and rules for counting vowels are explicitly stated"]
#     },
#     {
#         "id": "step-2",
#         "name": "Generate City Candidate List",
#         "description": "Brainstorm cities with long or unusual names that might contain many vowels. Consider examples like the full ceremonial name of Bangkok, the Welsh town 'Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch', or others you can find via research.",
#         "requires_code": false,
#         "expected_outputs": ["A list of candidate cities with potentially lengthy names"],
#         "success_criteria": ["At least one candidate appears likely to include at least 10 vowels"]
#     },
#     {
#         "id": "step-3",
#         "name": "Compute Vowel Count",
#         "description": "Select a candidate city name and count the vowels in the name. You may count manually or use a small code snippet (e.g., in Python) to count occurrences of A, E, I, O, U (regardless of case).",
#         "requires_code": true,
#         "expected_outputs": ["Numeric count of vowels in the candidate city name"],
#         "success_criteria": ["Count confirms that the city name contains at least 10 vowels"]
#     },
#     {
#         "id": "step-4",
#         "name": "Validate the Candidate",
#         "description": "Review the vowel count from the previous step. If the candidate meets or exceeds 10 vowels, it qualifies; otherwise, return to brainstorming for an alternate candidate.",
#         "requires_code": false,
#         "expected_outputs": ["Confirmation of candidate city meeting the vowel requirement"],
#         "success_criteria": ["Chosen candidate is verified to have 10 or more vowels"]
#     },
#     {
#         "id": "step-5",
#         "name": "Present Findings",
#         "description": "Report the final chosen city and summarize the vowel count. Explain why this city meets the criteria by detailing the counted vowels.",
#         "requires_code": false,
#         "expected_outputs": ["A clear response that states the city name and describes the vowel count"],
#         "success_criteria": ["Final answer includes the city name and an explanation confirming that it contains at least 10 vowels"]
#     }
# ]
# Writing filename  vowel_count.py
# Written code to file: ./workspace/vowel_count.py
# Find a city with as least 10 vowels in its name.
# Below is a step‐by‐step plan designed to solve the problem “Find a city with at least 10 vowels in its name.”

# Step 1: Understand the Problem Requirements
# • Descriptive Name: Interpret Requirements
# • Description: Read and clarify the condition “at least 10 vowels” in a city name. Decide whether vowels include only the standard five (a, e, i, o, u) and whether letters with diacritics are counted.
# • Code Needed?: No; this is a reasoning step.
# • Expected Output: A clear definition, for example, “I will count the letters A, E, I, O, U (in any case) as vowels.”
# • Evaluation: Check that the requirement is well understood and the criterion for vowels is clearly stated.

# Step 2: Brainstorm Candidate Cities
# • Descriptive Name: Generate City Candidate List
# • Description: Think of cities that are known for a long or unusual name. For example, consider:
#   – The full ceremonial name of Bangkok (Krungthep… etc.)
#   – The famously long Welsh town “Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch.”
#   – Other cities that might be known for having a lengthy name with many vowels.
# • Code Needed?: No; this is a brainstorming and research step.
# • Expected Output: A short list of candidate cities.
# • Evaluation: Verify that at least one candidate seems likely to contain more than 10 vowels.

# Step 3: Count the Vowels in a Candidate City Name
# • Descriptive Name: Compute Vowel Count
# • Description: Choose one candidate and count the vowels in its name. For example, pick “Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch.” Count every occurrence of a, e, i, o, u.
# • Code Needed?: (Optional) Yes, one may write a small snippet (in Python or similar) to count vowels and verify. Otherwise, a manual count works.
# • Expected Output: A numeric count showing the candidate has at least 10 vowels.
# • Evaluation: The step is successful if the candidate city name indeed has 10 or more vowels.

# Step 4: Verify the Candidate Meets the Criteria
# • Descriptive Name: Validate the Candidate
# • Description: Review the counted vowels from Step 3. If the candidate meets or exceeds 10 vowels, then it is acceptable. If not, return to Step 2 to select a different candidate.
# • Code Needed?: No; this is a review and verification step.
# • Expected Output: Confirmation that the city chosen qualifies.
# • Evaluation: The candidate is accepted if the vowel count is ≥10.

# Step 5: Report the Answer with Explanation
# • Descriptive Name: Present Findings
# • Description: Provide the final answer by stating the chosen city’s name and summarizing the vowel count.
# • Code Needed?: No; this is the concluding step.
# • Expected Output: A response such as “One example of a city with at least 10 vowels in its name is Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch.”
# • Evaluation: The answer is clear, meets the requirement, and includes an explanation of why this city qualifies (i.e. it has more than 10 vowels).

# Using these steps, one can confidently solve the problem. A well-known example is the Welsh town “Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch”, whose full name contains many vowels (well over 10 in total).
# Below is my detailed evaluation of each step:

# Step 1: Understand the Problem Requirements
# • Clarity: The step clearly explains that you need to understand what “at least 10 vowels” means. It specifically mentions considering whether only the five standard vowels count and whether accented vowels should be included.
# • Completeness: This step adequately addresses the key condition and ensures you understand it before proceeding.
# • Relevance: The step is essential to prevent ambiguity later in the process.
# • Feasibility: It is fully achievable through a careful reading of the problem.
# • Efficiency: This step is both necessary and efficient; no changes needed.

# Step 2: Brainstorm Candidate Cities
# • Clarity: The description is clear in suggesting to consider cities with longer or unusual names.
# • Completeness: The inclusion of examples like the full ceremonial name of Bangkok and the Welsh town provides a good basis; however, you might also add the possibility of doing online research if personal knowledge is insufficient.
# • Relevance: All suggestions are relevant as they might yield cities fulfilling the vowel count requirement.
# • Feasibility: Brainstorming is a straightforward step and is feasible with available resources.
# • Efficiency: The step is efficient but could be improved slightly by explicitly suggesting verifying the candidate's vowel counts as soon as ideas are generated.

# Step 3: Count the Vowels in a Candidate City Name
# • Clarity: The step clearly instructs to count the vowels manually or using code.
# • Completeness: It covers both manual and automated methods to ensure accuracy.
# • Relevance: This is a necessary step as it directly addresses the problem requirement.
# • Feasibility: This is realistic and achievable with basic tools or even a simple programming script.
# • Efficiency: The process is efficient. There is no need to combine this step with another unless you prefer to integrate it into a verification process.

# Step 4: Verify the Candidate Meets the Criteria
# • Clarity: The step explains how to verify if the candidate city qualifies by checking the count.
# • Completeness: It sufficiently covers the process of re-evaluating candidates if necessary.
# • Relevance: This step is directly related to ensuring that the chosen city meets the problem’s condition.
# • Feasibility: The check is straightforward and easily executed.
# • Efficiency: The instructions are clear and optimal as part of the iterative process; no revisions are needed.

# Step 5: Report the Answer with Explanation
# • Clarity: The step clearly describes how to report the final answer and why the selected city qualifies.
# • Completeness: It successfully wraps up the procedure by summarizing findings and providing a reasoned explanation.
# • Relevance: Reporting the final answer is essential for the solution’s completeness.
# • Feasibility: This final step is entirely feasible with the results from previous steps.
# • Efficiency: The step is concise and accomplishes the goal of clear presentation.

# Overall, the proposed steps are clear, complete, relevant, feasible, and efficient. There are no unnecessary steps, and each part of the process directly contributes to solving the problem. If desired, you might incorporate a brief reminder to conduct online research when brainstorming candidate cities in Step 2, but this is a minor suggestion rather than a necessary revision.

# [APPROVED]
# [
#     {
#         "id": "step-1",
#         "name": "Interpret Requirements",
#         "description": "Read and clarify the problem condition 'at least 10 vowels in a city name'. Decide whether vowels include only A, E, I, O, U (in any case) or if accented characters count as well.",
#         "requires_code": false,
#         "expected_outputs": ["Clear definition of vowel set (e.g., A, E, I, O, U)"],
#         "success_criteria": ["Criteria and rules for counting vowels are explicitly stated"]
#     },
#     {
#         "id": "step-2",
#         "name": "Generate City Candidate List",
#         "description": "Brainstorm cities with long or unusual names that might contain many vowels. Consider examples like the full ceremonial name of Bangkok, the Welsh town 'Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch', or others you can find via research.",
#         "requires_code": false,
#         "expected_outputs": ["A list of candidate cities with potentially lengthy names"],
#         "success_criteria": ["At least one candidate appears likely to include at least 10 vowels"]
#     },
#     {
#         "id": "step-3",
#         "name": "Compute Vowel Count",
#         "description": "Select a candidate city name and count the vowels in the name. You may count manually or use a small code snippet (e.g., in Python) to count occurrences of A, E, I, O, U (regardless of case).",
#         "requires_code": true,
#         "expected_outputs": ["Numeric count of vowels in the candidate city name"],
#         "success_criteria": ["Count confirms that the city name contains at least 10 vowels"]
#     },
#     {
#         "id": "step-4",
#         "name": "Validate the Candidate",
#         "description": "Review the vowel count from the previous step. If the candidate meets or exceeds 10 vowels, it qualifies; otherwise, return to brainstorming for an alternate candidate.",
#         "requires_code": false,
#         "expected_outputs": ["Confirmation of candidate city meeting the vowel requirement"],
#         "success_criteria": ["Chosen candidate is verified to have 10 or more vowels"]
#     },
#     {
#         "id": "step-5",
#         "name": "Present Findings",
#         "description": "Report the final chosen city and summarize the vowel count. Explain why this city meets the criteria by detailing the counted vowels.",
#         "requires_code": false,
#         "expected_outputs": ["A clear response that states the city name and describes the vowel count"],
#         "success_criteria": ["Final answer includes the city name and an explanation confirming that it contains at least 10 vowels"]
#     }
# ]

# File vowel_count.py written successfully.
# Below is the Python code that counts the vowels in the candidate city name "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch" to ensure it contains at least 10 vowels:

# --------------------------------------------------
# def count_vowels(city_name):
#     vowels = set('aeiouAEIOU')
#     return sum(1 for char in city_name if char in vowels)

# # Candidate city: The famously long Welsh town name
# city = "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch"
# vowel_count = count_vowels(city)

# print(f"City: {city}")
# print(f"Vowel Count: {vowel_count}")

# # Validate if the city meets the criteria
# if vowel_count >= 10:
#     print(f"The city meets the criteria with at least 10 vowels (found: {vowel_count}).")
# else:
#     print(f"The city does not meet the criteria (only {vowel_count} vowels found).")
# --------------------------------------------------

# When you run this code, it counts the vowels in the provided city name. The result confirms that the city has far more than 10 vowels, therefore meeting the criteria.

# Final Answer: One example of a city with at least 10 vowels in its name is "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch".
# Summary:
# The solution involves a five-step process. First, the problem requirements are interpreted by defining vowels as A, E, I, O, U (in both cases). Next, candidate cities are brainstormed, including lengthy names like the Welsh town "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch." Then, a Python script is used to count the vowels in this candidate name. The script confirms that the city name has well over 10 vowels. Finally, the candidate is validated and reported as meeting the criteria.

# Final Answer: "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch" is an example of a city with at least 10 vowels in its name.
