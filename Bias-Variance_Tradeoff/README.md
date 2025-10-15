# Demonstrating the Bias-Variance Tradeoff

This project provides a hands-on, empirical demonstration of the **bias-variance tradeoff**, a fundamental concept in machine learning.

Using the Telco Customer Churn dataset, we compare a simple linear model (Logistic Regression) with a complex ensemble model (XGBoost) to observe their behavior. The project shows how a model can be "wrong" in two different ways: by being too simple (high bias) or by being too complex (high variance). Finally, we tame the complex model using hyperparameter tuning to find a better balance and improve its generalization performance.

This repository also includes from-scratch implementations of **Logistic Regression** and a simplified **Gradient Boosting** algorithm to showcase a deeper understanding of the underlying mechanics.

-----

## \#\# The Core Concept: Bias-Variance Tradeoff 🧐

The total error of any supervised learning model can be decomposed into three components:

$Total\,Error = Bias^2 + Variance + Irreducible\,Error$

We can't control the irreducible error (random noise in the data), but we can often trade bias for variance.

  * **Bias (Underfitting)**: Bias is the error from a model's simplifying assumptions. A high-bias model is too simple and fails to capture the underlying patterns in the data. It consistently misses the mark.

      * **Analogy 🤔**: An archer who is consistent but always hits the top-left of the target. The aim is systematically wrong.
      * In this project, **Logistic Regression** represents our high-bias model.

  * **Variance (Overfitting)**: Variance is the error from a model's extreme sensitivity to the training data. A high-variance model is too complex and fits to the noise, not just the signal. It performs brilliantly on data it has seen but fails to generalize to new data.

      * **Analogy 🎯**: An archer whose shots are scattered all around the bullseye. The aim is correct on average, but each individual shot is unreliable.
      * In this project, the **default XGBoost Classifier** represents our high-variance model.

The goal is to find the "sweet spot" in model complexity that minimizes the total error on unseen data.

-----

## \#\# Project Workflow & Notebook Sections 📝

The project is structured in a Jupyter Notebook (`bv_tradeoff.ipynb`) that follows these steps:

### **1. Setup and Data Preprocessing**

  * **Libraries**: We import `pandas`, `scikit-learn`, and `xgboost`.
  * **Data Loading**: The Telco Customer Churn dataset is loaded from a CSV file.
  * **Data Cleaning**: Categorical features are one-hot encoded, and the binary target variable (`Churn`) is label encoded. The `customerID` column, which is not useful for modeling, is dropped before training.
  * **Train-Test Split**: The data is split into 80% for training and 20% for testing to ensure we can evaluate our models on unseen data.

### **2. Phase 1: Baseline Model Comparison**

To observe the bias-variance tradeoff, two baseline models are trained:

1.  **`LogisticRegression`**: A simple, linear model that is prone to high bias.
2.  **`XGBClassifier`**: A powerful, tree-based ensemble model that, with default parameters, is prone to high variance.

**Initial Findings**:
The results from this phase clearly show the tradeoff:

  * The **XGBoost** model achieves near-perfect scores on the training data but performs significantly worse on the test data. This large gap is a classic sign of **overfitting (high variance)**.
  * The **Logistic Regression** model has more modest scores, but its performance is very consistent between the training and test sets. This indicates **underfitting (high bias)**; the model is too simple to capture the full complexity of the data, but it generalizes well.

### **3. Phase 2: Taming High Variance with Hyperparameter Tuning**

The next step is to reduce the variance of the XGBoost model by tuning its hyperparameters. We use `GridSearchCV` with 5-fold cross-validation to find the optimal combination of parameters that control model complexity.

  * **Hyperparameter Grid**:
      * `max_depth`: Controls the maximum depth of each decision tree.
      * `n_estimators`: The number of trees in the ensemble.
      * `learning_rate`: A factor that shrinks the contribution of each tree.
  * **Evaluation**: The tuned XGBoost model is then evaluated on the test set to see if its generalization performance has improved.

### **4. Going Deeper: From-Scratch Implementations**

To demonstrate a fundamental understanding of the algorithms, the notebook also contains simple from-scratch implementations:

  * **`LogisticRegressionScratch`**: A class that implements logistic regression using gradient descent.
  * **`SimpleGradientBoosting`**: A class that implements a basic version of gradient boosting using decision stumps as weak learners.

These models are trained and evaluated on the same data to see how they perform compared to the optimized `scikit-learn` and `xgboost` library versions.

-----

## \#\# How to Run 🚀

1.  **Clone the repository**:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:

    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyterlab
    ```

4.  **Launch Jupyter Lab**:

    ```bash
    jupyter lab
    ```

5.  Open and run the `bv_tradeoff.ipynb` notebook.

-----

## \#\# Final Results Summary

The table below summarizes the performance of all models on the test set, showing the successful journey from diagnosing the bias-variance problem to solving it.

| Model                       | Test Accuracy | Test AUC-ROC | Analysis                                         |
| --------------------------- | :-----------: | :----------: | ------------------------------------------------ |
| Logistic Regression (sklearn) |    0.825    |    0.861     | **High Bias**: Stable but underfits the data.    |
| XGBoost (Default)           |    0.786    |    0.827     | **High Variance**: Overfits heavily to train data. |
| **XGBoost (Tuned)** |  **0.814** |  **0.862** | **Good Balance**: Reduced variance, best performance. |
| Logistic Regression (Scratch) |    0.797    |    0.816     | Proof of concept, decent performance.              |
| Simple Gradient Boosting (Scratch)|    0.735    |    0.838     | Proof of concept, shows boosting effect.         |

### **Conclusion**

The tuned XGBoost model achieved the highest AUC-ROC score on the test set, demonstrating that by carefully reducing its complexity, we were able to mitigate overfitting and find a robust model that generalizes well to new data. This project successfully highlights the practical steps involved in identifying and resolving the bias-variance tradeoff.