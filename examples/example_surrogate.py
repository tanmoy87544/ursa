import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import ExecutionAgent, PlanningAgent
from langchain_core.messages      import HumanMessage
from langchain_openai             import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama


problem = '''
Look for a file called finished_cases.csv in your workspace. If you find it, it should contain a column named something like "log Yield".

Write and execute a python file to:
  - Load that data into python.
  - Split the data into a training and test set.
  - Fit a Gaussian process model with gpytorch to the training data where "log Yield" is the output and the other variables are inputs.
  - Assess the quality of fit by r-squared on the test set and iterate if the current model is not good enough.
'''

def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        model = ChatOpenAI(
            model       = "o3-mini",
            max_tokens  = 10000,
            timeout     = None,
            max_retries = 2)
        # model = ChatOllama(
        #     model       = "llama3.1:8b",
        #     max_tokens  = 4000,
        #     timeout     = None,
        #     max_retries = 2
        # )
        
        init = {"messages": [HumanMessage(content=problem)]}
        
        print(f"\nSolving problem: {problem}\n")
        
        # Initialize the agent
        planner  = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)
        
        # Solve the problem
        planning_output = planner.action.invoke(init)
        print(planning_output["messages"][-1].content)
        final_results   = executor.action.invoke(planning_output)
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
    main()


#    {
#         "id": "step1_verify_data_file_exists",
#         "name": "Verify Data File Exists",
#         "description": "Check the current workspace for a file named 'finished_cases.csv'. Open the CSV file to inspect the header and confirm that it contains a column with a name similar to 'log Yield'. This may include handling any case or formatting variations in the header.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A confirmation message that 'finished_cases.csv' is found",
#             "A printed list of column names including one resembling 'log Yield'"
#         ],
#         "success_criteria": [
#             "The file exists in the workspace",
#             "The CSV header includes a column for 'log Yield' (or a close variation)"
#         ]
#     },
#     {
#         "id": "step2_load_and_preprocess",
#         "name": "Load Data and Pre-process",
#         "description": "Use pandas to load 'finished_cases.csv' into a DataFrame. Verify that the 'log Yield' column as well as all other columns (assumed to be input features) are loaded correctly. Perform preliminary exploratory data analysis including checking for missing values, high-level statistics, and, if necessary, apply any required data cleaning such as handling missing values or scaling the data.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A DataFrame with all columns loaded correctly",
#             "Printed output displaying the head and info of the DataFrame confirming the presence of 'log Yield'"
#         ],
#         "success_criteria": [
#             "The DataFrame loads without errors",
#             "The 'log Yield' column is present along with the expected input feature columns",
#             "Data quality issues (if any) are addressed as necessary"
#         ]
#     },
#     {
#         "id": "step3_data_splitting",
#         "name": "Data Splitting",
#         "description": "Separate the dataset into input features (all columns except 'log Yield') and the target variable ('log Yield'). Use scikit-learn's train_test_split to partition the data into training and test sets (commonly 80% training, 20% testing). Include a random_state parameter to ensure reproducibility.",
#         "requires_code": true,
#         "expected_outputs": [
#             "Four datasets: X_train, X_test, y_train, and y_test",
#             "Printed shapes of these datasets to verify the correct splits"
#         ],
#         "success_criteria": [
#             "The training and test sets have the expected sizes",
#             "The target variable is correctly separated from the input features"
#         ]
#     },
#     {
#         "id": "step4_construct_and_train_gp_model",
#         "name": "Construct and Train GP Model with gpytorch",
#         "description": "Define a Gaussian Process model using gpytorch. This involves setting up the model and likelihood with a basic kernel (e.g., RBF/Exponentiated Quadratic). Convert the training data (X_train, y_train) into PyTorch tensors. Implement the training routine including the optimizer and loss function, and ensure the model is in training mode during optimization. Log training output (loss values, iterations) and check for convergence.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A trained Gaussian Process model",
#             "Training logs displaying the loss decreasing over iterations"
#         ],
#         "success_criteria": [
#             "The model is trained without errors",
#             "Loss values decrease consistently over training iterations",
#             "The final model is ready for evaluation (with appropriate calls to model.eval() prior to prediction)"
#         ]
#     },
#     {
#         "id": "step5_model_evaluation",
#         "name": "Evaluate Model Performance",
#         "description": "Switch the GP model to evaluation mode and use it to predict the 'log Yield' for X_test. Extract the predictive means and compute the r-squared metric by comparing these predictions with y_test using scikit-learn's metrics. Optionally, produce visual diagnostics (e.g., scatter plots) to better assess model performance.",
#         "requires_code": true,
#         "expected_outputs": [
#             "Predicted values for the test set",
#             "A printed r-squared value indicating the model performance"
#         ],
#         "success_criteria": [
#             "The model successfully makes predictions on the test set",
#             "The r-squared metric is computed and printed",
#             "The performance is evaluated against predefined acceptance criteria"
#         ]
#     },
#     {
#         "id": "step6_iterate_model_improvements",
#         "name": "Model Iteration and Improvement",
#         "description": "If the r-squared performance does not meet the required threshold, iterate on the model improvements. This could involve trying different kernel functions or combinations, tuning hyperparameters such as the learning rate or the number of training iterations, or revisiting data preprocessing and feature engineering steps. Re-run the training and evaluation (Steps 4 and 5) with the revised settings to examine improvements in performance.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A revised version of the GP model with adjusted parameters",
#             "Improved r-squared performance from the test set predictions",
#             "Logging details that document changes and improvements in the model"
#         ],
#         "success_criteria": [
#             "The revised model shows a better r-squared score compared to the previous iteration",
#             "Adjustments provide clear insights into performance improvements",
#             "The iterative process is clearly documented and reproducible"
#         ]
#     }
# ]

# File gp_model_training.py written successfully.

# STDOUT: File 'finished_cases.csv' found.

# Columns in the CSV file: ['logYield', 'DT', 'Inner Shell', 'Tamper', 'Foam', 'Ablator']
#  and STDERR: Traceback (most recent call last):
#   File "/Users/mikegros/Projects/AIDI/lanl_scientific_agent/examples/workspace/gp_model_training.py", line 29, in <module>
#     raise ValueError("No column resembling 'log Yield' found in the CSV header.")
# ValueError: No column resembling 'log Yield' found in the CSV header.


# File gp_model_training.py written successfully.

# STDOUT: File 'finished_cases.csv' found.

# Columns in the CSV file: ['logYield', 'DT', 'Inner Shell', 'Tamper', 'Foam', 'Ablator']
# Target column identified as: 'logYield'

# Loading data from CSV...
# Data loaded. DataFrame shape: (484, 6)

# First 5 rows of the DataFrame:
#     logYield        DT  Inner Shell    Tamper      Foam   Ablator
# 0   5.000000  0.340000     0.100000  0.500000  0.420000  0.860000
# 1  12.256494  0.363024     0.184645  0.058616  0.785991  0.005090
# 2   5.000000  0.437481     0.179411  0.011461  0.385327  0.930655
# 3  12.838933  0.077483     0.291670  0.995889  0.649397  0.055405
# 4  13.307522  0.222572     0.772324  0.000000  0.000000  0.310098

# DataFrame info:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 484 entries, 0 to 483
# Data columns (total 6 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   logYield     484 non-null    float64
#  1   DT           484 non-null    float64
#  2   Inner Shell  484 non-null    float64
#  3   Tamper       484 non-null    float64
#  4   Foam         484 non-null    float64
#  5   Ablator      484 non-null    float64
# dtypes: float64(6)
# memory usage: 22.8 KB
# None

# Missing values in each column:
# logYield       0
# DT             0
# Inner Shell    0
# Tamper         0
# Foam           0
# Ablator        0
# dtype: int64
# After dropping missing values, new shape: (484, 6)

# Descriptive statistics:
#          logYield          DT  Inner Shell      Tamper        Foam     Ablator
# count  484.000000  484.000000   484.000000  484.000000  484.000000  484.000000
# mean    11.575568    0.312530     0.359337    0.429466    0.302593    0.358876
# std      4.462340    0.329263     0.332329    0.388934    0.321280    0.337251
# min      1.000000    0.000000     0.000000    0.000000    0.000000    0.000000
# 25%      5.000000    0.020542     0.064877    0.017512    0.042092    0.054947
# 50%     13.109814    0.205732     0.256065    0.351429    0.173660    0.232085
# 75%     14.687160    0.498364     0.606927    0.856828    0.462124    0.629069
# max     18.155115    1.000000     1.000000    1.000000    1.000000    1.000000
# Shapes after train-test split:
# X_train: (387, 5)
# X_test: (97, 5)
# y_train: (387,)
# y_test: (97,)
# Starting training...

# Iteration 1/100 - Loss: 4.942
# Iteration 10/100 - Loss: 3.350
# Iteration 20/100 - Loss: 2.745
# Iteration 30/100 - Loss: 2.521
# Iteration 40/100 - Loss: 2.416
# Iteration 50/100 - Loss: 2.358
# Iteration 60/100 - Loss: 2.319
# Iteration 70/100 - Loss: 2.291
# Iteration 80/100 - Loss: 2.270
# Iteration 90/100 - Loss: 2.254
# Iteration 100/100 - Loss: 2.240

# Training completed.

# R-squared on test set: 0.887

# The model performance (r-squared: 0.887) meets the expected criteria.
#  and STDERR: 2025-04-01 11:07:38.999 Python[74484:6947307] +[CATransaction synchronize] called within transaction

# The execution of the code was successful. Here's a brief summary of what happened:

# 1. The code verified that "finished_cases.csv" exists, and it printed the CSV file’s column names. It then normalized the column names to search for a column resembling "log Yield" and correctly identified "logYield" as the target column.

# 2. The dataset was loaded and basic exploratory analysis was performed. The code printed the head of the DataFrame, displayed information about the data types and missing values, dropped any missing rows (none in this case), and printed descriptive statistics.

# 3. The data was split into training and test sets (80% training and 20% testing). The shapes of X_train, X_test, y_train, and y_test were printed to confirm the split.

# 4. A Gaussian Process regression model was constructed using gpytorch. The training data was converted to PyTorch tensors, and the model was defined with a constant mean module and an RBF kernel inside a ScaleKernel. The model was trained for 100 iterations with Adam optimizer, and training logs were printed showing that the loss consistently decreased.

# 5. The model was switched to evaluation mode, and predictions on the test set were computed. The r-squared value was calculated using scikit-learn’s r2_score, resulting in an r-squared of 0.887, which indicates a very good amount of explained variance.

# 6. Finally, the code printed a message indicating that model performance meets the expected criteria (r-squared of 0.887 is above the threshold of 0.5).

# There was also an optional scatter plot provided for visual diagnostics. Overall, the process correctly loads, preprocesses, trains, and evaluates a GP model with gpytorch.
# The code executed successfully and met the objectives:

# • Step 1 confirmed that "finished_cases.csv" exists and identified "logYield" as the target column (using normalization to handle naming variations).  
# • Step 2 loaded the file into a pandas DataFrame, printed the head, info, and descriptive statistics, and checked for missing values. No missing values were found, and exploratory data analysis was performed.  
# • Step 3 split the data into input features (all columns except "logYield") and the target, using train_test_split with reproducibility (random_state=42).  
# • Step 4 converted the training data to PyTorch tensors, defined a GP regression model with a constant mean and RBF kernel, and trained it for 100 iterations while printing loss updates.  
# • Step 5 switched the model to evaluation mode, computed test predictions, calculated an r-squared value of 0.887, and optionally produced a scatter plot for visual diagnostics.  
# • Overall, the model performance meets the expected criteria, and recommendations for further iterations have been noted if necessary.

# This summary confirms that each step was properly implemented and executed with high performance.