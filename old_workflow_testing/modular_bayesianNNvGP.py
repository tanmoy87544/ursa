from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import HumanMessage
from langgraph.graph              import START, END, StateGraph


from .reflection import PlanningState
from .reflection import graph as planning_graph
from .execute    import app   as execution_graph


# llm = ChatOllama(
#     model       = "gemma3:12b",
#     # model       = "phi4",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

llm = ChatOpenAI(
    model       = "o1",
    max_tokens  = 4000,
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

Write and execute a python file to:
  - Load that data into python.
  - Split the data into a training and test set.
  - Fit both a Gaussian process model with gpytorch to the training data where "log Yield" is the output and the other variables are inputs.
      - Make sure that the number of training iterations are sufficient to converge
  - Fit a Bayesian neural network with numpyro to the same data.
      - Make sure that the number of training iterations are sufficient to converge
  - Assess the quality of fits by r-squared on the test set and summarize the quality of the Gaussian process against the neural network.
  - Assess the uncertainty quantification of the two models by coverage on the test set and with visualization.
'''

final_result = app.invoke({"messages": [HumanMessage( content=problem_string),],"reflection_steps":3},{"recursion_limit": 99})

# Outputs:
#
# Formalized [
#     {
#         "id": "step1",
#         "name": "File Discovery and Data Loading",
#         "description": "Search the current workspace for a file called finished_cases.csv. Once found, use Python (e.g., pandas) to load the CSV into a DataFrame and verify that it contains a column with a name similar to 'log Yield' (accounting for case or whitespace variations). Include error handling (try-except) to manage file-not-found issues gracefully.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A loaded DataFrame containing all the data from finished_cases.csv",
#             "A printed/logged confirmation that the DataFrame contains the target 'log Yield' column"
#         ],
#         "success_criteria": [
#             "The file is successfully found and read without errors",
#             "The DataFrame is inspected and verified to contain the target column",
#             "Graceful handling and messaging in case the file is not found"
#         ]
#     },
#     {
#         "id": "step2",
#         "name": "Data Preprocessing and Train-Test Splitting",
#         "description": "Examine the DataFrame for missing or corrupted entries, and perform any necessary cleaning. Define 'log Yield' as the target variable and treat all remaining columns as input features. Use a train_test_split (e.g., from scikit-learn) for an 80/20 train-test split while optionally standardizing/normalizing input features. Set a random seed for reproducibility.",
#         "requires_code": true,
#         "expected_outputs": [
#             "Two separate datasets: one for training and one for testing",
#             "Printed or logged summary statistics (shapes/dimensions) for both training and test sets"
#         ],
#         "success_criteria": [
#             "No overlapping data between training and testing sets",
#             "Correct separation of target and feature columns",
#             "Optional cleaning steps executed if missing or corrupted data was present"
#         ]
#     },
#     {
#         "id": "step3",
#         "name": "GP Model Training using gpytorch",
#         "description": "Define and implement a Gaussian process regression model in Python using the gpytorch library. Use the training data where 'log Yield' is the output and the rest are inputs. Configure the kernel (e.g., RBF kernel or appropriate alternative), likelihood, and optimizer, and run the training loop for enough iterations to guarantee convergence. Monitor training progress with loss logs and ensure that the model is switched to evaluation mode before prediction.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A trained Gaussian process model that has converged",
#             "Predicted 'log Yield' values for the test set along with uncertainty estimates (e.g., predictive variance)",
#             "Training loss graphs or logs showing a downward trend and stabilization"
#         ],
#         "success_criteria": [
#             "Training loss decreases and stabilizes over iterations",
#             "The model generates valid predictions along with uncertainty bands on the test set",
#             "Any convergence issues are remedied by adjusting iterations or learning rate"
#         ]
#     },
#     {
#         "id": "step4",
#         "name": "BNN Training using NumPyro",
#         "description": "Define a Bayesian neural network using NumPyro that takes the same input features and predicts 'log Yield' as the target. Set up the probabilistic model by defining priors for the network's weights and biases. Perform inference using either MCMC or stochastic variational inference (SVI), ensuring a sufficient number of iterations or samples for convergence. After training, generate posterior predictive samples for the test set and extract predictive means and variances.",
#         "requires_code": true,
#         "expected_outputs": [
#             "A trained Bayesian neural network model",
#             "Posterior predictive samples on the test set along with corresponding uncertainty estimates",
#             "Convergence diagnostics such as effective sample sizes, R-hat statistics (for MCMC) or ELBO loss profiles (for SVI)"
#         ],
#         "success_criteria": [
#             "The inference process converges appropriately as indicated by diagnostic metrics",
#             "Predictive samples display a reasonable spread in uncertainty estimates",
#             "Adjustments to hyperparameters or iteration count are made if convergence issues occur"
#         ]
#     },
#     {
#         "id": "step5",
#         "name": "Model Performance Evaluation – Accuracy (R-squared)",
#         "description": "Using the test set, evaluate both the GP and Bayesian neural network models by computing the R-squared value between the true 'log Yield' values and the predicted means. Display and compare the R-squared values side by side to determine which model fits the data better.",
#         "requires_code": true,
#         "expected_outputs": [
#             "Two R-squared scores (one for the GP model and one for the Bayesian neural network) printed or logged",
#             "A summary interpretation commenting on the relative performance based on these scores"
#         ],
#         "success_criteria": [
#             "R-squared values are computed accurately using a standard metric (e.g., sklearn.metrics.r2_score)",
#             "Output clearly distinguishes the performance of the two models",
#             "Any additional metrics (like RMSE) can be noted if necessary"
#         ]
#     },
#     {
#         "id": "step6",
#         "name": "Uncertainty Quantification Analysis and Visualization",
#         "description": "Assess the uncertainty quantification for both models on the test set. Calculate a coverage metric by determining the proportion of true 'log Yield' values that fall within the predictive intervals (e.g., ±2 standard deviations for a 95% confidence interval). Generate visualizations that plot the predicted means with uncertainty bands alongside the true values, and create calibration plots to compare predicted intervals with observed error frequencies. Summarize the findings in a printed/logged report.",
#         "requires_code": true,
#         "expected_outputs": [
#             "Numerical coverage percentages for both models",
#             "Plots that display predicted means, uncertainty bands, and a comparison against the true 'log Yield' values",
#             "A written summary comparing the uncertainty quantification performance of the GP and the Bayesian neural network"
#         ],
#         "success_criteria": [
#             "The coverage metric is calculated correctly and reflects model reliability",
#             "Visualizations clearly represent uncertainty and are easily interpretable",
#             "The summary provides a clear assessment of which model offers more reliable and well-calibrated uncertainty estimates"
#         ]
#     }
# ]
# Writing filename  pipeline_steps.py
# Written code to file: ./workspace/pipeline_steps.py
# RUNNING:  python pipeline_steps.py
# STDOUT:  
# STDERR:  Traceback (most recent call last):
#   File "/Users/mikegros/Projects/AIDI/workspace/pipeline_steps.py", line 19, in <module>
#     from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, Autoguide
# ImportError: cannot import name 'Autoguide' from 'numpyro.infer' (/Users/mikegros/envs/agentic/lib/python3.9/site-packages/numpyro/infer/__init__.py)

# Writing filename  pipeline_steps.py
# Written code to file: ./workspace/pipeline_steps.py
# RUNNING:  python pipeline_steps.py
# STDOUT:  Successfully loaded file from: finished_cases.csv
# Confirmed target column: 'logYield'
# Dropped 0 rows due to missing values.
# Initial training set shape: (387, 5)
# Initial test set shape:     (97, 5)

# ===== Training GP Model =====
# GP Training Iter 100/500, Loss: 4.6093
# GP Training Iter 200/500, Loss: 3.5577
# GP Training Iter 300/500, Loss: 3.1427
# GP Training Iter 400/500, Loss: 2.9228
# GP Training Iter 500/500, Loss: 2.7884

# ===== Training Bayesian Neural Network (NumPyro) =====

# STDERR:  Traceback (most recent call last):
#   File "/Users/mikegros/Projects/AIDI/workspace/pipeline_steps.py", line 361, in <module>
#     main()
#   File "/Users/mikegros/Projects/AIDI/workspace/pipeline_steps.py", line 321, in main
#     bnn_mcmc = train_bnn_mcmc(X_train_jnp, y_train_jnp, num_warmup=500, num_samples=500, num_chains=1, random_seed=42)
#   File "/Users/mikegros/Projects/AIDI/workspace/pipeline_steps.py", line 204, in train_bnn_mcmc
#     numpyro.seed(random_seed)
# AttributeError: module 'numpyro' has no attribute 'seed'

# Writing filename  pipeline_steps.py
# Written code to file: ./workspace/pipeline_steps.py
# RUNNING:  python pipeline_steps.py
# STDOUT:  Successfully loaded file from: finished_cases.csv
# Confirmed target column: 'logYield'
# Dropped 0 rows due to missing values.
# Initial training set shape: (387, 5)
# Initial test set shape:     (97, 5)

# ===== Training GP Model =====
# GP Training Iter 100/500, Loss: 4.6093
# GP Training Iter 200/500, Loss: 3.5577
# GP Training Iter 300/500, Loss: 3.1427
# GP Training Iter 400/500, Loss: 2.9228
# GP Training Iter 500/500, Loss: 2.7884

# ===== Training Bayesian Neural Network (NumPyro) =====

#                 mean       std    median      5.0%     95.0%     n_eff     r_hat
#      b1[0]     -0.02      1.37     -0.33     -1.90      2.32     47.22      1.04
#      b1[1]     -1.48      0.52     -1.54     -2.09     -0.86     38.47      1.04
#      b1[2]      0.23      1.33      0.31     -1.86      2.32     44.43      1.12
#      b1[3]     -0.02      1.69     -0.04     -2.53      2.58     33.56      1.02
#      b1[4]      0.02      1.42      0.24     -2.98      2.00     15.04      1.00
#      b1[5]      0.17      1.41      0.24     -2.22      2.52     22.03      1.13
#      b1[6]      0.07      1.53     -0.03     -2.23      2.62     36.60      1.01
#      b1[7]     -0.57      1.41     -0.43     -2.94      1.99     29.10      1.00
#      b1[8]     -0.29      1.15     -0.38     -1.89      1.52     44.90      1.05
#      b1[9]      0.22      1.22      0.45     -1.73      2.05     43.83      1.05
#     b1[10]      0.11      1.66      0.58     -2.68      2.25     27.38      1.01
#     b1[11]      0.07      1.40     -0.13     -1.98      2.58     46.51      1.00
#     b1[12]      0.34      1.57      0.54     -2.39      2.78     36.52      1.03
#     b1[13]     -0.09      1.51     -0.07     -2.74      2.17     19.65      1.04
#     b1[14]     -0.44      1.55     -0.60     -2.91      2.12     21.02      1.19
#     b1[15]      0.40      1.29      0.50     -1.83      2.47     55.16      1.01
#      b2[0]      3.66      0.98      3.67      2.29      5.41    322.44      1.00
#      sigma      1.39      0.06      1.39      1.29      1.49    267.59      1.00
#    w1[0,0]      0.08      0.84     -0.00     -1.10      1.48     33.79      1.00
#    w1[0,1]      1.66      0.45      1.71      1.02      2.22     29.93      1.05
#    w1[0,2]      0.08      0.89      0.01     -1.36      1.51    104.50      1.00
#    w1[0,3]      0.00      0.79      0.00     -1.34      1.16    139.20      1.00
#    w1[0,4]      0.59      0.67      0.78     -0.35      1.63     11.00      1.09
#    w1[0,5]      0.03      0.90     -0.04     -1.33      1.44     31.17      1.04
#    w1[0,6]     -0.01      0.81      0.05     -1.54      1.16    154.81      1.03
#    w1[0,7]     -0.27      0.85     -0.31     -1.54      1.01     19.76      1.10
#    w1[0,8]      0.10      0.85      0.05     -1.13      1.51    106.98      1.01
#    w1[0,9]     -0.18      0.87     -0.19     -1.66      1.06     61.46      1.01
#   w1[0,10]     -0.08      0.83      0.09     -1.89      0.90     24.16      1.01
#   w1[0,11]     -0.11      0.88     -0.16     -1.50      1.44    124.46      1.00
#   w1[0,12]     -0.07      0.86     -0.08     -1.59      1.21    135.10      1.00
#   w1[0,13]     -0.01      0.80      0.08     -1.44      1.07     33.19      1.01
#   w1[0,14]      0.10      0.83      0.08     -1.29      1.37     78.61      1.03
#   w1[0,15]     -0.19      0.93     -0.28     -1.47      1.40     29.06      1.06
#    w1[1,0]      0.06      0.79      0.02     -1.26      1.38     60.96      1.00
#    w1[1,1]      1.58      0.49      1.63      0.93      2.21     37.57      1.04
#    w1[1,2]      0.19      1.04      0.12     -1.30      2.11     63.87      1.00
#    w1[1,3]     -0.08      0.79     -0.09     -1.16      1.40     86.88      1.00
#    w1[1,4]      0.42      0.57      0.60     -0.38      1.17     13.14      1.04
#    w1[1,5]     -0.08      0.78     -0.08     -1.38      1.16    116.60      1.00
#    w1[1,6]     -0.12      0.80     -0.06     -1.56      1.12     76.91      1.00
#    w1[1,7]     -0.33      0.75     -0.31     -1.36      0.98     57.42      1.01
#    w1[1,8]     -0.16      0.80     -0.15     -1.58      1.09    148.41      1.00
#    w1[1,9]     -0.09      0.79     -0.14     -1.46      1.20     57.00      1.07
#   w1[1,10]     -0.03      0.89      0.10     -2.15      0.89     27.65      1.01
#   w1[1,11]     -0.04      0.76     -0.06     -1.02      1.42    125.01      1.01
#   w1[1,12]     -0.03      0.83     -0.05     -1.16      1.59    116.88      1.00
#   w1[1,13]      0.03      0.72      0.13     -1.24      0.88     32.83      1.04
#   w1[1,14]     -0.03      0.90     -0.04     -1.81      1.27    114.37      1.00
#   w1[1,15]     -0.04      0.85     -0.06     -1.36      1.36     64.82      1.01
#    w1[2,0]      0.01      0.96      0.05     -1.60      1.53     89.76      1.03
#    w1[2,1]      0.60      0.25      0.59      0.29      0.90    296.06      1.00
#    w1[2,2]      0.14      1.13      0.16     -1.76      2.10     32.80      1.06
#    w1[2,3]      0.04      0.76     -0.01     -1.21      1.27    239.08      1.00
#    w1[2,4]      0.33      0.58      0.46     -0.49      1.04     26.85      1.02
#    w1[2,5]      0.19      1.06      0.06     -1.18      2.31     26.29      1.07
#    w1[2,6]     -0.17      0.99     -0.10     -1.90      1.56     74.43      1.00
#    w1[2,7]     -0.23      0.86     -0.31     -1.76      1.16     74.09      1.01
#    w1[2,8]      0.05      0.97     -0.02     -1.30      1.76    152.89      1.01
#    w1[2,9]     -0.18      1.00     -0.16     -1.65      1.60    150.12      1.00
#   w1[2,10]      0.01      0.74     -0.04     -1.07      1.39    153.04      1.00
#   w1[2,11]     -0.06      1.07     -0.01     -2.35      1.46    126.58      1.02
#   w1[2,12]     -0.05      0.86     -0.03     -1.33      1.34    131.30      1.01
#   w1[2,13]     -0.04      0.73      0.01     -0.89      1.21    157.13      1.00
#   w1[2,14]      0.01      0.97     -0.00     -2.10      1.27     74.23      1.04
#   w1[2,15]     -0.13      1.08     -0.15     -2.05      1.79     50.78      1.06
#    w1[3,0]      0.17      1.40      0.10     -2.10      2.38     20.15      1.00
#    w1[3,1]      1.57      0.52      1.62      1.04      2.14     59.86      1.01
#    w1[3,2]      0.25      0.96      0.27     -1.47      1.58     71.47      1.03
#    w1[3,3]      0.06      0.95      0.01     -1.28      1.64     54.35      1.06
#    w1[3,4]      1.58      1.71      2.29     -1.48      3.58      8.46      1.02
#    w1[3,5]     -0.27      1.22     -0.26     -2.30      1.68     25.00      1.00
#    w1[3,6]     -0.09      1.15     -0.02     -2.22      1.47     60.54      1.00
#    w1[3,7]     -0.86      1.50     -0.52     -3.45      1.20      7.62      1.33
#    w1[3,8]     -0.07      1.12     -0.14     -1.87      1.81     45.53      1.01
#    w1[3,9]     -0.30      1.33     -0.17     -2.38      1.86     13.67      1.08
#   w1[3,10]     -0.05      1.12     -0.04     -1.81      1.56     32.11      1.00
#   w1[3,11]     -0.14      1.08     -0.21     -1.87      1.82     78.16      1.00
#   w1[3,12]     -0.05      1.15      0.07     -2.06      1.74     44.79      1.06
#   w1[3,13]     -0.15      1.41      0.15     -2.68      2.05     17.86      1.01
#   w1[3,14]      0.19      1.09      0.06     -1.40      2.19     52.75      1.00
#   w1[3,15]     -0.12      1.43     -0.10     -2.69      1.88     14.98      1.16
#    w1[4,0]     -0.25      1.11     -0.18     -2.69      1.20     22.80      1.08
#    w1[4,1]      1.09      0.48      1.18      0.70      1.76     18.74      1.08
#    w1[4,2]      0.24      0.89      0.21     -1.05      1.88     40.19      1.04
#    w1[4,3]     -0.17      1.10     -0.12     -1.88      1.90     26.01      1.08
#    w1[4,4]      0.36      0.94      0.69     -1.81      1.27     13.12      1.00
#    w1[4,5]      0.09      0.91      0.19     -1.41      1.66     44.12      1.01
#    w1[4,6]      0.29      1.10      0.15     -1.43      2.42     33.11      1.05
#    w1[4,7]     -0.17      0.91     -0.31     -1.17      1.77     32.66      1.12
#    w1[4,8]     -0.34      1.37     -0.20     -2.83      1.72     11.04      1.05
#    w1[4,9]      0.11      1.23      0.15     -2.34      2.03     40.87      1.00
#   w1[4,10]     -0.02      1.37     -0.19     -2.04      2.60     13.42      1.03
#   w1[4,11]     -0.32      1.20     -0.12     -2.79      1.22     15.21      1.00
#   w1[4,12]      0.18      0.90      0.19     -1.64      1.49     51.74      1.01
#   w1[4,13]      0.36      1.48      0.07     -1.55      3.12     11.42      1.09
#   w1[4,14]     -0.20      1.03     -0.10     -2.14      1.37     22.70      1.12
#   w1[4,15]      0.21      0.77      0.21     -0.93      1.52     92.25      1.00
#    w2[0,0]      0.52      1.68      0.49     -1.78      3.41     35.66      1.10
#    w2[1,0]     -1.98      0.98     -2.19     -3.13     -1.60     14.74      1.07
#    w2[2,0]     -0.08      1.19     -0.15     -1.72      1.82     17.28      1.06
#    w2[3,0]      0.07      2.03     -0.05     -3.06      3.55     33.69      1.00
#    w2[4,0]     -1.53      2.03     -2.30     -3.84      2.74     11.08      1.05
#    w2[5,0]     -0.17      1.67     -0.21     -3.33      2.41     17.94      1.09
#    w2[6,0]      0.10      1.83      0.26     -2.73      3.53     17.25      1.00
#    w2[7,0]      0.46      1.97      0.74     -2.77      3.40     18.00      1.13
#    w2[8,0]      0.21      1.51      0.17     -1.59      3.10     14.06      1.04
#    w2[9,0]     -0.39      1.50     -0.42     -2.70      2.61     27.33      1.00
#   w2[10,0]     -0.56      2.11     -0.57     -3.95      2.71     20.31      1.02
#   w2[11,0]      0.62      1.49      0.43     -1.46      3.09     25.99      1.06
#   w2[12,0]      0.07      1.81     -0.07     -2.28      3.74     22.68      1.08
#   w2[13,0]     -0.33      2.23     -0.28     -3.48      2.93     12.89      1.02
#   w2[14,0]      0.01      1.82      0.22     -3.36      2.82     22.77      1.01
#   w2[15,0]     -0.39      1.48     -0.54     -2.48      2.33     21.73      1.15

# Number of divergences: 0

# ===== Model Accuracy (R-squared) =====
# GP R-squared:  0.830
# BNN R-squared: 0.920

# ===== Uncertainty Coverage (Approx. 95% Interval) =====
# GP coverage:  92.78%
# BNN coverage: 97.94%

# Generating uncertainty plots...


# STDERR:  
#   0%|          | 0/1000 [00:00<?, ?it/s]
# warmup:   0%|          | 1/1000 [00:01<17:59,  1.08s/it, 1 steps of size 2.34e+00. acc. prob=0.00]
# warmup:   3%|▎         | 32/1000 [00:01<00:26, 36.54it/s, 1023 steps of size 1.41e-02. acc. prob=0.72]
# warmup:   6%|▌         | 58/1000 [00:01<00:13, 67.92it/s, 255 steps of size 1.93e-02. acc. prob=0.75] 
# warmup:   9%|▉         | 92/1000 [00:01<00:08, 113.23it/s, 127 steps of size 8.22e-03. acc. prob=0.76]
# warmup:  12%|█▏        | 118/1000 [00:01<00:06, 139.54it/s, 6 steps of size 7.41e-03. acc. prob=0.76] 
# warmup:  14%|█▍        | 144/1000 [00:01<00:05, 153.18it/s, 127 steps of size 1.48e-02. acc. prob=0.77]
# warmup:  17%|█▋        | 168/1000 [00:01<00:05, 152.65it/s, 127 steps of size 4.00e-02. acc. prob=0.77]
# warmup:  19%|█▉        | 194/1000 [00:01<00:04, 174.81it/s, 255 steps of size 3.71e-02. acc. prob=0.77]
# warmup:  22%|██▏       | 216/1000 [00:02<00:04, 158.07it/s, 255 steps of size 3.41e-02. acc. prob=0.78]
# warmup:  24%|██▎       | 236/1000 [00:02<00:04, 153.41it/s, 511 steps of size 2.87e-02. acc. prob=0.78]
# warmup:  25%|██▌       | 254/1000 [00:02<00:04, 150.97it/s, 1023 steps of size 3.92e-03. acc. prob=0.77]
# warmup:  27%|██▋       | 271/1000 [00:02<00:05, 142.92it/s, 255 steps of size 1.37e-02. acc. prob=0.78] 
# warmup:  29%|██▉       | 291/1000 [00:02<00:04, 155.16it/s, 255 steps of size 3.82e-02. acc. prob=0.78]
# warmup:  31%|███       | 308/1000 [00:02<00:04, 156.10it/s, 767 steps of size 2.43e-02. acc. prob=0.78]
# warmup:  33%|███▎      | 326/1000 [00:02<00:04, 161.13it/s, 511 steps of size 1.07e-02. acc. prob=0.78]
# warmup:  34%|███▍      | 344/1000 [00:02<00:03, 165.53it/s, 255 steps of size 3.02e-02. acc. prob=0.78]
# warmup:  36%|███▌      | 362/1000 [00:02<00:03, 168.33it/s, 127 steps of size 2.24e-02. acc. prob=0.78]
# warmup:  39%|███▊      | 387/1000 [00:03<00:03, 188.21it/s, 511 steps of size 1.38e-02. acc. prob=0.78]
# warmup:  41%|████      | 407/1000 [00:03<00:03, 187.09it/s, 255 steps of size 1.94e-02. acc. prob=0.78]
# warmup:  43%|████▎     | 430/1000 [00:03<00:02, 198.10it/s, 511 steps of size 1.62e-02. acc. prob=0.78]
# warmup:  45%|████▌     | 450/1000 [00:03<00:02, 197.47it/s, 511 steps of size 2.30e-02. acc. prob=0.78]
# warmup:  47%|████▋     | 471/1000 [00:03<00:02, 195.34it/s, 1023 steps of size 4.80e-03. acc. prob=0.78]
# warmup:  49%|████▉     | 492/1000 [00:03<00:02, 197.15it/s, 1023 steps of size 1.34e-02. acc. prob=0.78]
# sample:  51%|█████     | 512/1000 [00:03<00:02, 178.06it/s, 255 steps of size 1.51e-02. acc. prob=0.91] 
# sample:  53%|█████▎    | 531/1000 [00:03<00:02, 163.30it/s, 255 steps of size 1.51e-02. acc. prob=0.90]
# sample:  55%|█████▍    | 548/1000 [00:04<00:02, 160.54it/s, 255 steps of size 1.51e-02. acc. prob=0.89]
# sample:  57%|█████▋    | 567/1000 [00:04<00:02, 167.93it/s, 255 steps of size 1.51e-02. acc. prob=0.90]
# sample:  58%|█████▊    | 585/1000 [00:04<00:02, 161.37it/s, 1023 steps of size 1.51e-02. acc. prob=0.90]
# sample:  60%|██████    | 602/1000 [00:04<00:02, 160.86it/s, 255 steps of size 1.51e-02. acc. prob=0.91] 
# sample:  62%|██████▏   | 621/1000 [00:04<00:02, 168.28it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  64%|██████▍   | 645/1000 [00:04<00:01, 187.00it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  66%|██████▋   | 664/1000 [00:04<00:01, 174.29it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  68%|██████▊   | 682/1000 [00:04<00:01, 173.91it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  70%|███████   | 700/1000 [00:04<00:01, 171.85it/s, 767 steps of size 1.51e-02. acc. prob=0.91]
# sample:  72%|███████▏  | 719/1000 [00:04<00:01, 176.45it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  74%|███████▎  | 737/1000 [00:05<00:01, 172.85it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  76%|███████▌  | 757/1000 [00:05<00:01, 177.17it/s, 767 steps of size 1.51e-02. acc. prob=0.91]
# sample:  78%|███████▊  | 775/1000 [00:05<00:01, 155.10it/s, 767 steps of size 1.51e-02. acc. prob=0.91]
# sample:  79%|███████▉  | 792/1000 [00:05<00:01, 155.03it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  81%|████████  | 808/1000 [00:05<00:01, 154.02it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  83%|████████▎ | 827/1000 [00:05<00:01, 161.23it/s, 767 steps of size 1.51e-02. acc. prob=0.91]
# sample:  84%|████████▍ | 844/1000 [00:05<00:00, 157.01it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  86%|████████▌ | 862/1000 [00:05<00:00, 159.10it/s, 1023 steps of size 1.51e-02. acc. prob=0.91]
# sample:  88%|████████▊ | 879/1000 [00:06<00:00, 152.35it/s, 511 steps of size 1.51e-02. acc. prob=0.91] 
# sample:  90%|████████▉ | 895/1000 [00:06<00:00, 153.32it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  91%|█████████▏| 913/1000 [00:06<00:00, 157.56it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  93%|█████████▎| 932/1000 [00:06<00:00, 166.28it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  95%|█████████▍| 949/1000 [00:06<00:00, 163.54it/s, 255 steps of size 1.51e-02. acc. prob=0.91]
# sample:  97%|█████████▋| 967/1000 [00:06<00:00, 165.49it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# sample:  98%|█████████▊| 984/1000 [00:06<00:00, 158.38it/s, 1023 steps of size 1.51e-02. acc. prob=0.91]
# sample: 100%|██████████| 1000/1000 [00:06<00:00, 147.81it/s, 511 steps of size 1.51e-02. acc. prob=0.91]
# 2025-03-20 11:34:48.474 Python[53462:1437634] +[CATransaction synchronize] called within transaction
# 2025-03-20 11:35:08.290 Python[53462:1437634] +[CATransaction synchronize] called within transaction
