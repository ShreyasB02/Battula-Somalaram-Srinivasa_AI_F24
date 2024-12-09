# **Feature Selection using Genetic Algorithm with Parallel Processing**  
*A Case Study on the Wisconsin Breast Cancer Dataset*
---
## **Overview**
This project leverages **Genetic Algorithms (GAs)**, an evolutionary optimization technique, to perform feature selection on the **Wisconsin Breast Cancer Dataset**. By integrating **parallel processing**, the framework accelerates computation, making it scalable and efficient for large datasets. The selected features aim to improve model performance while maintaining interpretability.

---

## **Motivation**
Feature selection is a crucial step in machine learning pipelines. It helps:
- Enhance model accuracy.
- Reduce overfitting.
- Simplify model interpretability.

Traditional methods like Recursive Feature Elimination (RFE) and Mutual Information are limited by their greedy search mechanisms. In contrast, this project explores Genetic Algorithms to identify optimal feature subsets by evolving solutions iteratively.

---

## **Key Features**
1. **Genetic Algorithm**: Optimizes feature subsets using population initialization, selection, crossover, and mutation.
2. **Parallel Processing**: Uses Python's `multiprocessing` library to evaluate fitness scores concurrently, reducing runtime by up to **60%**.
3. **Metric Flexibility**: Supports multiple metrics (accuracy, precision, recall, F1-score) to align with diverse objectives.
4. **Dynamic Visualization**: Tracks performance improvements across generations, e.g., accuracy vs. number of features.
5. **Validation**: Compares GA-selected features with traditional methods (e.g., RFE) and Random Forest feature importance.

---

## **Dataset**
The **Wisconsin Breast Cancer Dataset** contains:
- **569 samples**, categorized as benign (B) or malignant (M).
- **30 numerical features**, including radius, texture, and perimeter.
- Class label: Diagnosis (`B` or `M`).

This dataset is widely used for binary classification in healthcare, making it ideal for evaluating feature selection techniques.

---

## **Methodology**
1. **Population Initialization**: Generate a population of chromosomes representing feature subsets.
2. **Fitness Evaluation**: Evaluate chromosomes using model performance metrics.
3. **Selection**: Retain top-performing chromosomes for reproduction.
4. **Crossover**: Combine parent chromosomes to create offspring.
5. **Mutation**: Introduce diversity by flipping random bits in chromosomes.
6. **Early Stopping**: Stop the algorithm if no improvement is observed over a predefined number of generations.
7. **Validation**: Compare selected features with baseline methods and domain knowledge.

---

## **Results**
1. **Performance**:
   - Model accuracy improved to **96.5%** with GA-selected features, compared to **95.8%** using all features.
   - Precision, recall, and F1-score also showed minor improvements.
2. **Efficiency**:
   - Parallel processing reduced runtime by approximately **60%**.
3. **Feature Overlap**:
   - **75% overlap** was observed between GA-selected features and features identified as important by Random Forest.

---

Here’s the updated **usage section** for the `README.md` based on your provided code:

---

## **Usage**

### **Step 1: Prepare the Dataset**
Ensure the input dataset (`new_data.csv`) is available in the folder. The dataset should contain feature columns and a target column (`label`).

### **Step 2: Run the Genetic Algorithm**
Execute the script to perform feature selection using Genetic Algorithms:

```bash
python genetic_algorithm.py
```

### **Detailed Steps in the Code**
#### **Data Loading and Preprocessing**
- The dataset is loaded from `new_data.csv`.
- Features and the target label are split using the `split()` function:
   ```python
   X_train, X_test, Y_train, Y_test = split(features, label)
   ```

#### **Run Genetic Algorithm**
- The Genetic Algorithm evolves feature subsets over multiple generations:
   ```python
   best_chromo, best_scores = generations_with_metrics(
       df, 
       label, 
       X_train, 
       X_test, 
       Y_train, 
       Y_test, 
       size=20, 
       n_feat=X_train.shape[1], 
       n_parents=10, 
       mutation_rate=0.1, 
       n_gen=50, 
       model=LogisticRegression(max_iter=1000), 
       metrics=["accuracy", "precision", "recall", "f1"]
   )
   ```

#### **Visualize Progress**
- Plot the performance of the algorithm across generations:
   ```python
   plot_generation_progress(best_scores["accuracy"], metric_name="Accuracy")
   ```

#### **Save Results**
- Save the selected features for each generation:
   ```python
   save_results(best_chromo, feature_names=features.columns, filename="best_features.txt")
   ```

### **Outputs**
1. **Performance Plot**:
   - A plot showing how the accuracy evolves across generations is saved as `performance_plot.png`.

2. **Selected Features**:
   - A file `best_features.txt` lists the selected features for each generation.

---


### **Sample Output**
1. **Best Chromosome**:
   Displays the most optimized feature subset.
2. **Metrics for Each Generation**:
   Outputs metrics such as accuracy, precision, recall, and F1-score for each generation.
3. **Selected Features**:
   ```plaintext
   Generation 1 Best Features: radius_mean, texture_mean, perimeter_mean
   Generation 2 Best Features: radius_mean, texture_mean, smoothness_mean
   ...
   ```

---


## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - `numpy` and `pandas` for data manipulation.
  - `sklearn` for model training and metrics.
  - `multiprocessing` for parallel processing.
  - `matplotlib` and `seaborn` for visualization.

---


## **Conclusion**
This project demonstrates the effectiveness of Genetic Algorithms for feature selection, enhanced with parallel processing and metric flexibility. The results validate the importance of the selected features and highlight the framework's applicability to real-world datasets like Wisconsin Breast Cancer.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
Special thanks to:
- The UCI Machine Learning Repository for providing the Wisconsin Breast Cancer Dataset.

