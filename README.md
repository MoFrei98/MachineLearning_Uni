# Machine Learning Iris Classification Pipeline

A Python-based machine learning project that classifies iris flowers into three species using Support Vector Machine (SVM) models with an object-oriented architecture.

## Project Overview

This project implements a complete machine learning pipeline for the Iris dataset. It demonstrates key OOP principles including:
- **Design Patterns**: Factory Pattern, Strategy Pattern, Polymorphism
- **Abstract Base Classes**: Metric interface for extensibility
- **Separation of Concerns**: Modular components for data, models, training, and evaluation

### Training Goal

Classify iris flowers into 3 different species based on 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

### Iris Species to Classify

The three iris flower species (gattungen) to distinguish are:

1. **Iris setosa** - Setosa iris (small petals, short flowers)
2. **Iris versicolor** - Versicolor iris (medium-sized flowers)
3. **Iris virginica** - Virginia iris (large petals, tall flowers)

The model learns to differentiate these species based on the 4 morphological features.

## Project Structure

```
PythonPortfolio/
├── launcher.py                 # Main interactive console application
├── script.py                   # Legacy standalone script example
├── ml/                         # Machine learning module
│   ├── dataset.py              # Data loading and splitting
│   ├── trainer.py              # Training orchestration
│   ├── kernel.py               # Kernel base class
│   ├── metrics/
│   │   ├── metric.py           # Abstract metric base class
│   │   ├── accuracy.py         # Accuracy metric implementation
│   │   └── confusion_matrix.py # Confusion matrix metric
│   └── models/
│       ├── model.py            # Abstract model base class
│       ├── model_factory.py    # Factory for model creation
│       └── svm_model.py        # SVM model implementation
└── doku/                       # Documentation and concepts
```

## Key Components

### 1. **Dataset** (`ml/dataset.py`)
Handles data loading and preprocessing:
- Loads the Iris dataset from seaborn
- Splits data into training and test sets
- Supports configurable test size and random state for reproducibility

```python
ds = Dataset(test_size=0.3, random_state=42)
ds.load_data()
x_train, y_train = ds.get_train_data()
x_test, y_test = ds.get_test_data()
```

### 2. **Models** (`ml/models/`)
- **Model**: Abstract base class defining the interface
- **SVMModel**: Concrete implementation using scikit-learn's SVC
- **ModelFactory**: Factory pattern for creating different model variants

#### Currently Implemented Models

**Support Vector Machine (SVM) Variants:**
- `svm_linear` - Linear kernel (best for linearly separable data)
- `svm_rbf` - Radial Basis Function kernel (default, handles non-linear patterns)
- `svm_poly` - Polynomial kernel with degree=3 (captures polynomial relationships)
- `svm_sigmoid` - Sigmoid kernel (similar to neural network activation)

#### Planned Models (Future Implementation)

- **Decision Tree** - Tree-based classifier with feature importance
- **Random Forest** - Ensemble of decision trees for robust predictions
- **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
- **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
- **Logistic Regression** - Linear model for binary/multiclass classification
- **Neural Network / MLP** - Multi-layer perceptron for deep learning
- **Gradient Boosting** - XGBoost/LightGBM for state-of-the-art performance

All models follow the same `Model` interface for seamless integration with the pipeline.

### 3. **Metrics** (`ml/metrics/`)
- **Metric**: Abstract base class for all metrics
- **Accuracy**: Calculates classification accuracy
- **ConfusionMatrix**: Generates detailed confusion matrix with statistics

Extensible design allows easy addition of new metrics.

### 4. **Trainer** (`ml/trainer.py`)
Orchestrates the training and evaluation pipeline:
- Accepts model, dataset, and metrics
- Trains the model on training data
- Evaluates on test data using all defined metrics
- Uses polymorphism to calculate metrics

### 5. **Kernel** (`ml/kernel.py`)
Base class for kernel implementations (foundation for future extensibility).

## Usage

### Interactive Mode (Recommended)

Run the interactive console application:

```bash
python launcher.py
```

The program will guide you through:
1. **Model Selection**: Choose from 4 SVM variants
2. **Hyperparameters**: Set test size, epochs, and random state
3. **Training**: Automatic data loading, model creation, and training
4. **Evaluation**: View accuracy and confusion matrix
5. **Loop Control**: Run multiple training sessions without restarting

**Example Session:**
```
--- Starting Machine Learning Pipeline ---

--- Available Models ---
1. svm_linear
2. svm_rbf
3. svm_poly
4. svm_sigmoid

Select a model (1-4): 2
Enter test size (0.0-1.0, default: 0.3): 0.3
Enter number of epochs (default: 1): 1
Enter random state for reproducibility (default: 42): 42

✓ Configuration:
  Model: svm_rbf
  Test size: 0.3
  Epochs: 1
  Random state: 42

1. Loading data...
✓ Data loaded successfully.

2. Creating model...
✓ Model 'svm_rbf' created.

3. Initializing metrics...
✓ 2 metrics initialized.

4. Initializing trainer...
5. Executing training...
Training completed.

6. Performing evaluation...

--- Evaluation Results ---

Accuracy:
0.9333...

Confusion Matrix:
...
```

### Legacy Script Mode

For reference, `script.py` demonstrates a standalone implementation without OOP architecture:

```bash
python script.py
```

## Requirements

### Dependencies

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

### Python Version

Python 3.10+ (uses match/case statements)

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the launcher:
   ```bash
   python launcher.py
   ```

## Design Patterns Used

### 1. **Factory Pattern**
`ModelFactory` creates different SVM models based on algorithm type:
```python
model = ModelFactory.get_model("svm_rbf", input_shape=4, output_shape=3)
```

### 2. **Strategy Pattern**
Different kernels (linear, rbf, poly, sigmoid) are strategies for the SVM model.

### 3. **Polymorphism**
Metrics implement a common `Metric` interface:
```python
for metric in metrics_list:
    result = metric.calculate(y_true, y_pred)
```

### 4. **Template Method**
`Trainer` orchestrates the training workflow using model and metric interfaces.

## Configuration

### Hyperparameters

- **Test Size**: Proportion of data for testing (default: 0.3)
- **Epochs**: Number of training iterations (default: 1)
- **Random State**: Seed for reproducibility (default: 42)
- **Kernel**: SVM kernel type (linear, rbf, poly, sigmoid)

All parameters are customizable via the interactive console.

## Output Metrics

### Accuracy
Simple accuracy score: correct predictions / total predictions

### Confusion Matrix
Detailed breakdown per class:
- Correct predictions on diagonal
- Incorrect predictions off-diagonal
- Shows per-class error rates

## Future Enhancements

### Planned Model Implementations
- [ ] **Decision Tree Classifier** - Tree-based classification with feature importance
- [ ] **Random Forest** - Ensemble learning combining multiple decision trees
- [ ] **K-Nearest Neighbors (KNN)** - Instance-based learning with distance metrics
- [ ] **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
- [ ] **Logistic Regression** - Linear model for multiclass classification
- [ ] **Neural Network (MLP)** - Multi-layer perceptron for deep learning
- [ ] **Gradient Boosting (XGBoost/LightGBM)** - State-of-the-art ensemble methods

### Infrastructure & Features
- [ ] Separate kernel classes with base class (see TODO in model_factory.py)
- [ ] Cross-validation support
- [ ] Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Model persistence (save/load trained models)
- [ ] Visualization of decision boundaries
- [ ] Feature importance analysis
- [ ] Support for multiple datasets
- [ ] Benchmarking suite for model comparison

## Code Quality Features

✅ **Type hints** (partial implementation)  
✅ **Docstrings** for all classes and functions  
✅ **Exception handling** in input validation  
✅ **Modular design** for easy testing and extension  
✅ **English comments and output** for international collaboration  
✅ **Consistent naming conventions**

## Testing

To verify the setup works:

```bash
python launcher.py
# Select model: 2 (svm_rbf)
# Press Enter for all prompts to use defaults
# View evaluation results
```

Expected accuracy: ~93% on Iris dataset with default parameters.

## Author

Created as a portfolio project demonstrating OOP principles and machine learning workflow in Python.

## License

This project is provided as-is for educational purposes.

## References

- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)

