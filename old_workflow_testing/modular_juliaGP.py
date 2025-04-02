from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage
from langgraph.graph              import START, END, StateGraph


from reflection import PlanningState
from reflection import graph as planning_graph
from execute    import app   as execution_graph


# llm = ChatOllama(
#     model       = "gemma3:12b",
#     # model       = "phi4",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

llm = ChatOpenAI(
    model       = "o1",
    max_tokens  = 10000,
    timeout     = None,
    max_retries = 2
)

 
workflow = StateGraph(PlanningState)

workflow.add_node("planning", planning_graph)
workflow.add_node("action",  execution_graph)


workflow.add_edge(START, "planning")
workflow.add_edge("planning", "action")
workflow.add_edge("action", END)

app = workflow.compile()

# problem_string = "Find a city with as least 10 vowels in its name."
problem_string = '''
Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

Write and execute a Julia code to:
  - Load that data into Julia.
  - Split the data into a training and test set.
  - Fit a Gaussian process model to the training data where "log Yield" is the output and the other variables are inputs.
      - Make sure that the number of training iterations are sufficient to converge
  - Assess the quality of fits by r-squared on the test set and summarize the quality of the Gaussian process.
  - Assess the uncertainty quantification of the model by coverage on the test set and with visualization.
  - Ensure Julia packages are installed as necessary.
'''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],"reflection_steps":3},{"recursion_limit": 99})

# Formalized [
#   {
#     "id": "step_1",
#     "name": "Environment and Package Setup",
#     "description": "Check for the existence of 'finished_cases.csv' in the workspace using Julia’s isfile function. Install and import necessary packages such as CSV.jl, DataFrames.jl, GaussianProcesses.jl, StatsBase.jl, and Plots.jl. Include conditional routines to install any missing packages and provide a printed message regarding file existence.",
#     "requires_code": true,
#     "expected_outputs": [
#       "Printed message confirming whether 'finished_cases.csv' exists.",
#       "Successful run of installation commands (if required) and successful 'using' statements."
#     ],
#     "success_criteria": [
#       "The file existence message is printed correctly.",
#       "All required packages are installed and imported without errors."
#     ]
#   },
#   {
#     "id": "step_2",
#     "name": "Data Loading",
#     "description": "Load 'finished_cases.csv' using CSV.jl and convert it into a DataFrame with DataFrames.jl. Display the first few rows (using head) of the DataFrame to confirm that the data, including the 'log Yield' column and predictor variables, is loaded correctly.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A DataFrame printed to the console containing a 'log Yield' column and other predictive columns."
#     ],
#     "success_criteria": [
#       "The DataFrame displays correctly and includes the necessary columns."
#     ]
#   },
#   {
#     "id": "step_3",
#     "name": "Data Preprocessing and Cleaning",
#     "description": "Verify that the column 'log Yield' (or a close variant) exists. Check the data for missing or anomalous entries and apply necessary cleaning steps such as renaming columns or casting data types appropriately. Optionally, consider scaling or normalizing predictor variables if required by the GP model.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A printed summary using functions like describe() that confirms data integrity and shows presence of the 'log Yield' column.",
#       "Optional printout of any modifications performed (e.g., renaming columns, handling missing values)."
#     ],
#     "success_criteria": [
#       "The 'log Yield' column is correctly identified and cleaned.",
#       "The data is in proper format for model training."
#     ]
#   },
#   {
#     "id": "step_4",
#     "name": "Data Splitting",
#     "description": "Randomly split the dataset into training (70%) and test sets (30%) using a package like StatsBase.jl or built-in routines, ensuring reproducibility by setting a fixed random seed. Print out the dimensions of both sets to verify the split.",
#     "requires_code": true,
#     "expected_outputs": [
#       "Two DataFrames: one for training and one for testing.",
#       "Printed dimensions of each DataFrame showing a 70/30 split."
#     ],
#     "success_criteria": [
#       "The training and test sets are created with the correct size ratio.",
#       "Random seed is set to ensure reproducibility."
#     ]
#   },
#   {
#     "id": "step_5",
#     "name": "Gaussian Process Model Training",
#     "description": "Fit a Gaussian process model on the training set. Define 'log Yield' as the output and the remaining columns as inputs. Specify an appropriate kernel (e.g., squared exponential) and mean function (e.g., zero or constant) and set a large number of training iterations to ensure convergence by using the optimization features of GaussianProcesses.jl. Print training status or log details indicating the convergence progress.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A trained Gaussian process model object.",
#       "Logs or printed messages showing the optimization iterations and convergence."
#     ],
#     "success_criteria": [
#       "The GP model trains without errors.",
#       "The optimization logs suggest convergence and the model object is ready for making predictions."
#     ]
#   },
#   {
#     "id": "step_6",
#     "name": "Predictive Quality Evaluation using r-squared",
#     "description": "Use the trained GP model to predict 'log Yield' for the test set inputs. Calculate the r-squared value (coefficient of determination) comparing predictions against true values, using the formula r² = 1 – (SS_res/SS_tot). Print out the r-squared value to summarize model performance.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A printed r-squared value (e.g., 'Test set r-squared: 0.85') summarizing the model fit."
#     ],
#     "success_criteria": [
#       "The r-squared value is computed correctly.",
#       "The printed r-squared value indicates the predictive quality of the GP model."
#     ]
#   },
#   {
#     "id": "step_7",
#     "name": "Uncertainty Quantification Assessment",
#     "description": "Calculate prediction intervals (e.g., 95% confidence intervals) from the GP model predictions by using the predicted means and variances. Assess uncertainty quantification quality by determining the proportion of true 'log Yield' values in the test set that fall within these intervals. Print this coverage metric.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A printed coverage metric (e.g., 'Coverage: 93%') indicating the proportion of true values within the predictive intervals."
#     ],
#     "success_criteria": [
#       "The prediction intervals are constructed adequately from the GP model.",
#       "The computed coverage is close to the nominal confidence level, confirming reasonable uncertainty quantification."
#     ]
#   },
#   {
#     "id": "step_8",
#     "name": "Visualization of Results and Uncertainty",
#     "description": "Create plots to visualize the GP model predictions versus the observed 'log Yield' values, including error bars or shaded regions to display the uncertainty intervals. Optionally, generate a plot of the residuals distribution to further assess model performance.",
#     "requires_code": true,
#     "expected_outputs": [
#       "One or more plots showing predicted versus true values with uncertainty intervals (error bars or shaded areas).",
#       "Optional residual plot illustrating the distribution of prediction errors."
#     ],
#     "success_criteria": [
#       "The visualizations are rendered correctly.",
#       "The plots clearly communicate both the model's predictive performance and the associated uncertainties."
#     ]
#   },
#   {
#     "id": "step_9",
#     "name": "Final Summary and Documentation",
#     "description": "Output a summary report that includes the r-squared value, uncertainty coverage metric, and key insights from the visualizations. Ensure the code contains clear comments explaining each significant block and the overall methodology to facilitate reproducibility and understanding.",
#     "requires_code": true,
#     "expected_outputs": [
#       "A printed summary of findings detailing the model performance and uncertainty assessment.",
#       "Well-commented code that documents each step of the process."
#     ],
#     "success_criteria": [
#       "The final summary is clear, concise, and accurately documents the outcomes of the analysis.",
#       "The code is well documented and offers clarity on implementation procedures."
#     ]
#   }
# ]
# Writing filename  gaussian_process_analysis.jl
# Written code to file: ./workspace/gaussian_process_analysis.jl
# RUNNING:  julia gaussian_process_analysis.jl
# STDOUT:  Step 1: Environment and Package Setup

# STDERR:  ERROR: LoadError: syntax: "import" expression not at top level
# Stacktrace:
#  [1] top-level scope
#    @ ~/Projects/AIDI/lanl_scientific_agent/workflows/workspace/gaussian_process_analysis.jl:25
# in expression starting at /Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/workspace/gaussian_process_analysis.jl:25

# Writing filename  gaussian_process_analysis.jl
# Written code to file: ./workspace/gaussian_process_analysis.jl
# RUNNING:  julia gaussian_process_analysis.jl
# STDOUT:  Step 1: Environment and Package Setup
# [INFO] Found the file 'finished_cases.csv'. Proceeding...

# Step 2: Data Loading
# [INFO] Successfully loaded data. Preview:
# 5×6 DataFrame
#  Row │ logYield  DT         Inner Shell  Tamper     Foam      Ablator
#      │ Float64   Float64    Float64      Float64    Float64   Float64
# ─────┼───────────────────────────────────────────────────────────────────
#    1 │   5.0     0.34          0.1       0.5        0.42      0.86
#    2 │  12.2565  0.363024      0.184645  0.0586164  0.785991  0.00508962
#    3 │   5.0     0.437481      0.179411  0.011461   0.385327  0.930655
#    4 │  12.8389  0.0774829     0.29167   0.995889   0.649397  0.055405
#    5 │  13.3075  0.222572      0.772324  0.0        0.0       0.310098

# Step 3: Data Preprocessing and Cleaning

# STDERR:  ERROR: LoadError: UndefVarError: `df` not defined in `Main`
# Suggestion: check for spelling errors or missing imports.
# Hint: a global variable of this name may be made accessible by importing Rmath in the current active module Main
# Stacktrace:
#  [1] top-level scope
#    @ ~/Projects/AIDI/lanl_scientific_agent/workflows/workspace/gaussian_process_analysis.jl:75
# in expression starting at /Users/mikegros/Projects/AIDI/lanl_scientific_agent/workflows/workspace/gaussian_process_analysis.jl:75

# Writing filename  gaussian_process_analysis.jl
# Written code to file: ./workspace/gaussian_process_analysis.jl
# Writing filename  gaussian_process_analysis.jl
# Written code to file: ./workspace/gaussian_process_analysis.jl