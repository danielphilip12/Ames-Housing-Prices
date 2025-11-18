# 🏡 Ames Housing Data Project
Welcome to our next project! It's time to start exploring and modeling. 🚀

#### Refrence
The Ames Housing dataset was compiled by Dean De Cock in 2011, for use in data science education.

* * * * *

### Part 1: Exploratory Data Analysis (EDA) and Data Cleaning 🧑‍🔬

#### Primary Learning Objectives:

-   Understand the Ames Housing dataset through EDA. 📊
-   Perform any necessary data cleaning and feature engineering. 🧹
-   Prepare a clean dataset for modeling. ✔️

#### Exploring the Dataset

You will start by exploring the Ames Housing dataset, which contains over 70 features related to the properties in Ames, Iowa. Your goal is to identify relationships, correlations, and any potential outliers that could affect the prediction model later on.

In the **EDA.ipynb** notebook, you will:

-   Load and inspect the **ameshousing.csv** file. 🔍
-   Review the data dictionary [here](data/DataDocumentation.txt) to understand the meaning of each feature in the dataset. This is crucial for making informed decisions during your analysis.
-   Conduct Exploratory Data Analysis (EDA) by:
    -   Analyzing distributions of key features. 📈
    -   Identifying missing data and handling it appropriately (e.g., filling or removing). ❓
    -   Identifying correlations between features and the target (sale price). 🔗
    -   Visualizing relationships between variables using plots (e.g., histograms, scatter plots, heatmaps). 🎨
    -   Identifying and handling outliers. ⚠️

#### Feature Engineering:

-   Create new features or transform existing ones if necessary. 🔄

#### Data Cleaning:

-   Handle missing values and remove or impute outliers for the **ameshousing.csv**.
-   Apply the same cleaning methods consistently to the entire dataset to ensure a clean version for modeling, without dropping any rows.

At the end of your EDA, you should have a clean dataset ready for modeling.

#### Saving the Cleaned Dataset

Once you have completed the EDA and data cleaning, save the cleaned dataframe as a new CSV file to be used for the next step.\
Save your work in **EDA.ipynb**. 💾

```python
# Example code to save cleaned dataset
dataset_df.to_csv('cleaned_ameshousing.csv')
```

* * * * *

### Part 2: Modeling Process 📈

Once you have completed your EDA and saved your cleaned dataset, you will now build a regression model in the **modeling.ipynb** notebook.

#### Set-Up 🛠️

Before you begin working on the modeling process, complete the following steps:

-   Ensure you have a clean dataset (`cleaned_ameshousing.csv`) from the EDA step.
-   Review the material on the regression modeling process.

#### The Modeling Process

In **modeling.ipynb**, you will:

-   Load the **cleaned_ameshousing.csv** file. 📥
-   Use the cleaned data to create and refine a regression model.
-   Split your data into training and testing sets. 🔄
-   Perform feature selection if necessary. ⚙️
-   Train the model using Linear Regression. 📊
-   Evaluate the performance using metrics like RMSE or R². 📈
-   Make predictions on the cleaned dataset.

#### Model Evaluation, Enhancement, and Iteration

-   Document any improvements or enhancements made to the model throughout the process. This includes any changes in features, parameters, or model adjustments.
-   Track the decisions you make to improve model performance, such as how you addressed overfitting, underfitting, or performance bottlenecks.
-   Evaluate your model performance using appropriate metrics and adjust your model as needed. Consider:
    -   How your model generalizes to new data.
    -   Iterating on the model by tuning hyperparameters or refining features.

* * * * *

#### Final Deliverables:

-   **Technical Report**: Create your own **README.md** file for this project.
-   **Notebooks**: Submit the **EDA.ipynb** and **modeling.ipynb** notebooks.
-   **Cleaned Dataset**: Submit the **cleaned_ameshousing.csv** file.
-   **Presentation**: A 5-10 minute presentation detailing your findings, model performance, and key recommendations.

Be sure to **document your model improvements and enhancements** in your notebooks, including the reasoning behind any decisions you made to refine your model.

## 📝 Rubric
The grading for this project will be based on three primary categories: **Functional**, **Interpersonal**, and **Technical**. Each category includes specific dimensions that will be evaluated on a scale from 1 to 5. To pass the project, you must score **3 or higher** in each component.

| FIT Category | Dimension | Exemplary (5) | Proficient (4) | Minimally Career-Ready (3) | Needs Improvement (2) | Incomplete Work (1) |
| --- | --- | --- | --- | --- | --- | --- |
| Functional | Value/Impact | Deep, impactful insights with clear application to housing market strategies and decisions. | Actionable insights with strong relevance to housing market decisions. | Some meaningful insights, but lacks strong application to the domain. | Minimal insights with unclear relevance to the housing market. | No meaningful insights or model predictions. |
|  | Requirements | Fully detailed steps, with deep insights and consistency throughout. | All key steps thoroughly completed with full attention to detail. | All key steps present but lacks detail or consistency in some parts. | Missing some steps or minimal effort in parts of the process. | Missing key steps, incomplete work. |
|  | Timelines | Ahead of schedule, with strong planning and time management. | On time with proper time management. | On time but rushed or incomplete. | Late with minimal explanation. | Significantly late without explanation. |
| Interpersonal | Professionalism | Highly polished, engaging, and professional throughout. | Clear, professional, and well-organized. | Professional tone, but lacks clarity or polish in some parts. | Somewhat professional but lacking clarity in sections. | Unprofessional tone, unclear, and poorly structured. |
|  | Presentation | Highly engaging, clear, and visually appealing with strong storytelling. | Well-organized, engaging, and visually appealing presentation. | Clear presentation, but lacks some visual appeal or engagement. | Somewhat unclear and lacking visual organization. | Disorganized and unclear presentation. |
|  | Feedback | Fully integrates feedback, demonstrating a strong learning curve and improvements. | Actively engages with feedback and refines the work accordingly. | Incorporates some feedback with moderate refinement. | Minimal engagement or slight changes based on feedback. | No engagement with feedback. |
| Technical | Complexity | Advanced and deep exploration with advanced feature engineering techniques. | Thorough EDA with comprehensive cleaning and innovative feature engineering. | Solid exploration, cleaning, and feature engineering with some complexity. | Some exploration and cleaning, but limited feature engineering. | Very basic or incomplete analysis, minimal feature engineering. |
|  | Design | Exceptionally well-organized notebooks with clear and thorough explanations and commentary. | Well-organized notebooks with clear structure, comments, and explanations. | Clear and organized notebooks but lacking some explanations or comments in some parts. | Somewhat organized notebooks with minimal clarity in explanations. | Poorly structured notebooks, unclear code. |
|  | Reliability | Reliable, highly accurate model with clear explanations of data handling and robust predictions. | Accurate model with correct handling of outliers, missing data, and well-structured predictions. | Generally correct model performance, but lacks some consistency or clarity. | Some errors in model predictions, outliers, or missing data handling. | Numerous errors in data cleaning or model performance. |


**Passing Criteria:**

*   **Minimum Requirement**: A score of **3 or higher** in each component to pass.
*   **Total Points/Percentages**: These are provided for informational purposes but do not affect the passing criteria.

* * *