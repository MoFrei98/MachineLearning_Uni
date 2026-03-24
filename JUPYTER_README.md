# 🌸 Flower Classification - Machine Learning Portfolio

A comprehensive machine learning pipeline for iris flower classification using 8 different algorithms.

## 📊 Overview

This project implements a complete ML pipeline that trains and evaluates multiple machine learning models on the famous Iris dataset. The architecture demonstrates key software engineering principles like the **Strategy Pattern**, **Factory Pattern**, and **Polymorphism**.

**Goal:** Classify iris flowers into 3 species based on 4 botanical features.

---

## 🚀 Quick Start with Jupyter Notebook

### 1. Clone/Open the Project

```bash
cd C:\Users\morri\PycharmProjects\PythonPortfolio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
pip install -e .
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open **`Example_ML_Pipeline.ipynb`** in your browser.

---

## 📚 Available Models

The project includes **8 different ML algorithms**:

### SVM Variants (4 models)
1. **svm_linear** - Linear kernel for linearly separable data
2. **svm_rbf** - RBF kernel for non-linear patterns (Gaussian)
3. **svm_poly** - Polynomial kernel with degree 3
4. **svm_sigmoid** - Sigmoid kernel (neural network-like)

### Other Algorithms (4 models)
5. **decision_tree** - Tree-based classifier (max_depth=5)
6. **knn** - K-Nearest Neighbors (k=5)
7. **random_forest** - Ensemble of decision trees (n_estimators=100)
8. **lda** - Linear Discriminant Analysis

---

## 🏗️ Project Structure

```
PythonPortfolio/
├── setup.py                          # Package configuration for pip install
├── requirements.txt                  # Dependencies (numpy, pandas, sklearn, jupyter, etc.)
├── launcher.py                       # Command-line interface
├── Example_ML_Pipeline.ipynb        # Jupyter notebook with examples
│
├── ml/                              # Main package
│   ├── __init__.py                 # Package initialization
│   ├── dataset.py                  # Data loading and splitting
│   ├── trainer.py                  # Model training orchestration
│   │
│   ├── models/                     # ML Models
│   │   ├── __init__.py
│   │   ├── model.py               # Abstract base class
│   │   ├── model_factory.py       # Factory pattern implementation
│   │   ├── svm_model.py          # Support Vector Machine
│   │   ├── tree_model.py         # Decision Tree
│   │   ├── knn_model.py          # K-Nearest Neighbors
│   │   ├── randomforest_model.py # Random Forest
│   │   └── lda_model.py          # Linear Discriminant Analysis
│   │
│   ├── kernel/                     # SVM Kernels
│   │   ├── __init__.py
│   │   ├── kernel.py             # Abstract base class
│   │   ├── linear_kernel.py      # Linear kernel
│   │   ├── rbf_kernel.py         # RBF kernel
│   │   ├── poly_kernel.py        # Polynomial kernel
│   │   └── sigmoid_kernel.py     # Sigmoid kernel
│   │
│   └── metrics/                    # Evaluation metrics
│       ├── __init__.py
│       ├── metric.py              # Abstract base class
│       ├── accuracy.py            # Accuracy metric
│       └── confusion_matrix.py    # Confusion matrix

```

---

## 📖 Usage Examples

### Via Command Line (launcher.py)

```bash
python launcher.py
```

Interactive menu to:
- Select a model
- Configure hyperparameters (test_size, epochs, random_state)
- Train and evaluate
- Loop for multiple experiments

### Via Python Script

```python
from ml.dataset import Dataset
from ml.models.model_factory import ModelFactory
from ml.trainer import Trainer
from ml.metrics.accuracy import Accuracy

# 1. Load data
dataset = Dataset(test_size=0.3, random_state=42)
dataset.load_data()

# 2. Create model
model = ModelFactory.get_model("svm_rbf", input_shape=4, output_shape=3)

# 3. Train
trainer = Trainer(model=model, dataset=dataset, metrics_list=[Accuracy()])
trainer.train()

# 4. Evaluate
results = trainer.evaluate()
print(f"Accuracy: {results['Accuracy']:.4f}")
```

### Via Jupyter Notebook

Open `Example_ML_Pipeline.ipynb` and run the cells sequentially:

1. **Setup**: Install dependencies
2. **Load Data**: Load and explore Iris dataset
3. **Visualize**: Show feature distributions
4. **Train**: Train all 8 models
5. **Compare**: Visualize performance comparison
6. **Analyze**: Deep dive into the best model
7. **Predict**: Make predictions on new data

---

## 🎯 Key Design Patterns

### 1. **Factory Pattern** (`ModelFactory`)
Centralized model creation with a registry:
```python
model = ModelFactory.get_model("svm_rbf", input_shape=4, output_shape=3)
```

### 2. **Strategy Pattern** (`Kernel` & `Metric`)
Interchangeable algorithms without changing the core code:
```python
kernel = RBFKernel()
model.build(kernel=kernel)
```

### 3. **Template Method Pattern** (`Trainer`)
Orchestrates the ML pipeline in a consistent manner.

### 4. **Polymorphism**
- Abstract classes: `Model`, `Kernel`, `Metric`
- Concrete implementations: `SVMModel`, `LinearKernel`, `Accuracy`

---

## 📊 Dataset Information

**Iris Dataset:**
- **Samples:** 150 flowers
- **Classes:** 3 species (setosa, versicolor, virginica)
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Train/Test Split:** Configurable (default: 70% train, 30% test)

---

## 🔧 Configuration & Hyperparameters

### In `launcher.py` or Jupyter:

```python
# Dataset configuration
test_size = 0.3          # 30% for testing
random_state = 42       # For reproducibility
epochs = 1              # Training iterations

# Model defaults (in ModelFactory):
"svm_rbf":      RBFKernel()
"decision_tree": max_depth=5
"knn":          n_neighbors=5
"random_forest": n_estimators=100
"lda":          default params
```

---

## 📈 Performance Tips

1. **Normalize/Scale Features**: Add StandardScaler before training
2. **Try Different Kernels**: svm_linear, svm_rbf, svm_poly
3. **Cross-Validation**: Implement k-fold CV for better estimates
4. **Hyperparameter Tuning**: Use GridSearchCV for optimal params
5. **Ensemble Methods**: Combine multiple models

---

## ✅ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computing |
| pandas | >=2.0.0 | Data manipulation |
| seaborn | >=0.12.0 | Data visualization |
| matplotlib | >=3.7.0 | Plotting |
| scikit-learn | >=1.3.0 | ML algorithms |
| jupyter | >=1.0.0 | Notebook environment |
| ipykernel | >=6.25.0 | Jupyter kernel support |

---

## 🧪 Testing

Run the Python files to verify everything works:

```bash
python -m py_compile ml/models/*.py ml/kernel/*.py ml/metrics/*.py launcher.py
```

Or run the launcher:

```bash
python launcher.py
```

---

## 📝 TODOs & Future Improvements

- [ ] Implement cross-validation
- [ ] Add feature scaling/normalization
- [ ] Create hyperparameter tuning grid
- [ ] Add ensemble voting classifier
- [ ] Implement ROC/AUC metrics
- [ ] Add data visualization utilities
- [ ] Create model persistence (save/load)
- [ ] Add logging framework

---

## 📄 License

This is an educational project for a portfolio/coursework.

---

## 👤 Author

Created as part of a Machine Learning course portfolio.

---

## 📞 Support

For issues or questions:
1. Check `Example_ML_Pipeline.ipynb` for working examples
2. Review the code comments in `launcher.py`
3. Inspect model implementations in `ml/models/`

---

**Happy Machine Learning! 🚀**

