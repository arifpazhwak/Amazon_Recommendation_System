# Amazon Product Recommendation System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square" alt="Python 3.x"/>
  <img src="https://img.shields.io/badge/Pandas-Used-green.svg?style=flat-square" alt="Pandas Used"/>
  <img src="https://img.shields.io/badge/NumPy-Used-green.svg?style=flat-square" alt="NumPy Used"/>
  <img src="https://img.shields.io/badge/Surprise-Used-orange.svg?style=flat-square" alt="Surprise Used"/>
  <img src="https://img.shields.io/badge/Plotly-Used-purple.svg?style=flat-square" alt="Plotly Used"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT"/>
</p>

---

## Project Overview

This project focuses on building and evaluating various recommendation system models using the Amazon product reviews dataset for electronics. The primary goal is to recommend products to customers based on their past rating behaviors. This project was undertaken as part of my coursework for the **MIT Data Science and Machine Learning Program**. While the problem and dataset are common in educational settings, this particular notebook, including its analysis, model implementation, and interpretation, was completed independently.

---

## Business Context

In today's information-rich environment, consumers face an overwhelming number of choices, often leading to decision fatigue. Recommender Systems are vital tools that help businesses guide consumers by providing personalized and relevant product suggestions. E-commerce giants like Amazon invest heavily in developing sophisticated recommendation algorithms to enhance user engagement, improve customer experience, and drive sales. Amazon's system, known for its accuracy, intelligently analyzes customer preferences to offer tailored recommendations, with item-to-item collaborative filtering being one of its foundational techniques.

---

## Project Objective

As a data scientist, the objective of this project is to develop a recommendation system for Amazon's electronics category. Using a dataset of customer ratings, the aim is to:
1.  Extract meaningful insights from the user-item interaction data.
2.  Build and evaluate several types of recommendation models, including popularity-based, collaborative filtering (User-User and Item-Item), and matrix factorization (SVD) approaches.
3.  Tune hyperparameters to optimize model performance.
4.  Provide a final recommendation for the most effective model based on comprehensive evaluation metrics.

---
## Dataset

The project utilizes the `ratings_Electronics.csv` dataset, which contains customer ratings for various electronic products.
* **Attributes:** `userId` (unique identifier for each user), `productId` (unique identifier for each product), `Rating` (rating given by the user to the product, on a scale of 1-5).
* **Original Size:** The raw dataset contains 7,824,482 ratings.
* **Processed Size:** After preprocessing (filtering users with <50 ratings and items with <5 ratings), the dataset used for modeling comprised 65,290 ratings from 1,540 users for 5,689 products.
* **Download:** Due to its size (approx. 318MB), the raw dataset is not included in this repository. You can download it from Google Drive:
    [Download ratings_Electronics.csv](https://drive.google.com/uc?export=download&id=11IpdaNRnzvMHMbBIFEstrSe4EkOiQyJC)
    *(Please download and place it in the project's root directory or update the path in the notebook.)*

---
## Methodology

This project followed a structured approach to building and evaluating recommendation systems:

1.  **Data Loading & Initial Inspection:** Loaded the dataset and examined its structure, data types, and basic statistics.
2.  **Data Cleaning & Preprocessing:**
    * Dropped the `timestamp` column as it was not required for this analysis.
    * Filtered the dataset to manage computational resources and improve model robustness by retaining users who provided at least 50 ratings and items that received at least 5 ratings.
3.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of ratings.
    * Determined the number of unique users and products in the filtered dataset.
    * Identified the most active users.
    * Utilized Plotly for interactive visualizations.
4.  **Model Building & Evaluation:**
    * **Rank-Based (Popularity) Recommender:** Implemented as a baseline, recommending items based on average ratings and interaction counts.
    * **User-User Collaborative Filtering (KNNBasic):**
        * Built a baseline model using cosine similarity.
        * Performed hyperparameter tuning (for `k` and similarity metrics) using `GridSearchCV`.
        * Evaluated both baseline and tuned models.
    * **Item-Item Collaborative Filtering (KNNBasic):**
        * Built a baseline model using cosine similarity.
        * Performed hyperparameter tuning (for `k`, `min_k`, and similarity metrics) using `GridSearchCV` with professor-guided parameters.
        * Evaluated both baseline and tuned models.
    * **Matrix Factorization (SVD):**
        * Built a baseline model with default parameters.
        * Performed hyperparameter tuning (for `n_epochs`, `lr_all`, `reg_all`) using `GridSearchCV`.
        * Evaluated both baseline and tuned models.
5.  **Evaluation Metrics:** Assessed models using Root Mean Squared Error (RMSE) for rating prediction accuracy, and Precision@10, Recall@10, and F1-score@10 for top-N recommendation quality (with a relevance threshold of 3.5).

---
## Key Findings & Model Performance Summary

After building and evaluating various models, the following key performance metrics were observed on the held-out test set:

| Model                                            | RMSE        | Precision@10 | Recall@10 | F1-score@10 |
| :----------------------------------------------- | :---------- | :----------- | :-------- | :---------- |
| User-User CF (Baseline, k=40, cosine)            | 1.0012      | 0.855        | 0.858     | 0.856       |
| User-User CF (Tuned, k=50, cosine)               | 1.0012      | 0.856        | 0.858     | 0.857       |
| Item-Item CF (Baseline, k=40, cosine)            | 0.9950      | 0.838        | 0.845     | 0.841       |
| Item-Item CF (Tuned, k=30, min_k=6, msd)         | 0.9576      | 0.839        | 0.880     | 0.859       |
| SVD (Baseline, default params)                   | 0.8894      | 0.849        | 0.877     | **0.863** |
| **SVD (Tuned, n_epochs=30, lr=0.005, reg=0.04)** | **0.8887** | **0.853** | 0.871     | 0.862       |

The **Tuned SVD model** demonstrated the best overall performance, achieving the lowest RMSE (0.8887) and very strong Precision@10 (0.853) and F1-score@10 (0.862). While the Tuned Item-Item model showed the highest Recall@10 (0.880), the SVD models offered a superior balance across all metrics.

---
## Final Model Recommendation & Conclusion

Based on the comprehensive evaluation, the **Tuned SVD model** (with parameters: `n_epochs=30`, `lr_all=0.005`, `reg_all=0.04`) is recommended as the most effective model for this dataset and task. It provides the best rating prediction accuracy and excellent top-N recommendation quality. This highlights the strength of matrix factorization techniques in capturing latent user preferences and item characteristics.

This project successfully demonstrates the end-to-end process of building, evaluating, and optimizing various recommendation algorithms, providing valuable insights into their comparative performances.

---
## Tools and Libraries Used

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Surprise:** A Python scikit for building and analyzing recommender systems (used for KNNBasic, SVD, GridSearchCV, evaluation metrics).
* **Plotly (Express & Graph Objects):** For creating interactive visualizations.
* **Matplotlib & Seaborn:** (Mentioned as initially loaded, though Plotly was primarily used for final visuals).

---
## File Structure

* `01_Amazon_Recommendation_System.ipynb`: The main Jupyter Notebook containing all Python code, detailed analysis, visualizations, and step-by-step interpretations.
* `02_ratings_Electronics.csv`: The raw dataset (to be downloaded separately, see Dataset section).
* `03_Amazon_Recommendation_System.html`: An HTML export of the Jupyter Notebook for easy viewing.
* `README.md`: This file, providing an overview of the project.

---
## How to Use/Reproduce

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/Amazon_Recommendation_System.git](https://github.com/YourGitHubUsername/Amazon_Recommendation_System.git)
    cd Amazon_Recommendation_System
    ```
    (Replace `YourGitHubUsername` with your actual GitHub username)
2.  **Download the Dataset:** Download `ratings_Electronics.csv` from the Google Drive link provided in the [Dataset](#dataset) section and place it in the root directory of the cloned project.
3.  **Set up a Python Environment:** It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-surprise plotly matplotlib seaborn jupyter
    ```
    (Or provide a `requirements.txt` file for easier installation: `pip install -r requirements.txt`)
5.  **Run the Jupyter Notebook:**
    Open and run the `01_Amazon_Recommendation_System.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, VS Code with Jupyter extension).

---
*Author: Arif Pazhwak*
