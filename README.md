# 🌸 Machine Learning Iris Flower Classification Pipeline

A comprehensive machine learning project for iris flower classification using 8 different algorithms with object-oriented architecture and design patterns.

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [How to Use](#how-to-use)
4. [References](#references)

---

## 📊 Project Overview

This project implements a complete machine learning pipeline for the **Iris dataset**. It demonstrates key software engineering principles:

- **Design Patterns**: Factory Pattern, Strategy Pattern, Polymorphism
- **Abstract Base Classes**: Extensible interfaces for models and metrics
- **Separation of Concerns**: Modular components for data, models, training, and evaluation
- **Python Best Practices**: Type hints, docstrings, exception handling, English documentation

### Training Goal

Classify iris flowers into **3 different species** based on **4 botanical features**:
- Sepal length
- Sepal width
- Petal length
- Petal width

### Iris Species to Classify

1. **Iris setosa** - Small petals, short flowers
2. **Iris versicolor** - Medium-sized flowers
3. **Iris virginica** - Large petals, tall flowers

### Implemented Models

#### SVM Variants (4 models)

| Model | Kernel | Use Case |
|-------|--------|----------|
| `svm_linear` | Linear | Linearly separable data |
| `svm_rbf` | RBF (Gaussian) | Non-linear patterns (default) |
| `svm_poly` | Polynomial (degree=3) | Polynomial relationships |
| `svm_sigmoid` | Sigmoid | Similar to neural network activation |

#### Other Algorithms (4 models)

| Model | Type | Use Case |
|-------|------|----------|
| `decision_tree` | Tree-based | Interpretable decisions (max_depth=5) |
| `knn` | Instance-based | Local patterns (k=5) |
| `random_forest` | Ensemble | Robust predictions (n_estimators=100) |
| `lda` | Linear model | Linear class separation |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (project uses match/case statements)
- **pip** (Python package manager)

### Installation Steps

#### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\morri\PycharmProjects\PythonPortfolio
```

#### Step 2: Install Dependencies

**Option A: Install via requirements.txt (Simple)**

```powershell
pip install -r requirements.txt
```

**Option B: Install as editable package (Recommended)**

```powershell
pip install -e .
```

This approach allows you to:
- Import the `ml` package from anywhere
- Automatically install all dependencies
- Use the project in Jupyter Notebook and scripts

#### Step 3: Verify Installation

```powershell
python -c "from ml.dataset import Dataset; print('✓ Installation successful!')"
```

---

## 🎯 How to Use

### Option 1: Interactive Command-Line Interface

Run the launcher with interactive menu:

```powershell
python launcher.py
```

**Features:**
1. ✅ **Model Selection** - Choose from 8 different algorithms
2. ✅ **Hyperparameter Configuration** - Set test size, epochs, random state
3. ✅ **Automatic Training** - Data loading, model creation, training
4. ✅ **Evaluation** - View accuracy and confusion matrix
5. ✅ **Loop Mode** - Run multiple training sessions without restarting

---

### Option 2: Jupyter Notebook

Ideal for interactive exploration, visualization, and experimentation.

#### Start Jupyter Notebook

```powershell
jupyter notebook
```

This opens Jupyter in your default browser at `http://localhost:8888`

#### Using the Example Notebook

1. In Jupyter, navigate to and open **`Example_ML_Pipeline.ipynb`**
2. Run all cells sequentially using **Cell** → **Run All**
3. Or run cells individually using **Shift + Enter**

#### Useful Jupyter Commands

```python
# Enable auto-reload of modules (useful during development)
%load_ext autoreload
%autoreload 2

# Show plots inline
%matplotlib inline

# Adjust plot size
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

# Time code execution
%timeit some_function()

# Run shell commands
!pip list
!dir
```

---

### Option 3: Direct Python Scripting

For standalone scripts and automation use script.py
Run with:

```powershell
python script.py
```

---

## 📊 Output & Metrics

### Accuracy

Simple accuracy metric:

```
Accuracy = Correct Predictions / Total Predictions
Expected: ~93% on Iris dataset with default parameters
```

### Confusion Matrix

Example output for confusion matrix:

```
setosa       -> Corret: 48 | Incorrect:  0
versicolor   -> Corret: 42 | Incorrect:  2
virginica    -> Corret: 39 | Incorrect:  4
```

---

## 🚀 Quick Commands Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run interactive CLI | `python launcher.py` |
| Start Jupyter | `jupyter notebook` |
| Verify installation | `python -c "from ml.dataset import Dataset; print('✓ OK')"` |
| Run example script | `python example_script.py` |
| List available models | `python -c "from ml.models.model_factory import ModelFactory; print(list(ModelFactory._registry.keys()))"` |

---

## 📚 References

- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Design Patterns](https://refactoring.guru/design-patterns/python)
- [Python Best Practices](https://pep8.org/)

