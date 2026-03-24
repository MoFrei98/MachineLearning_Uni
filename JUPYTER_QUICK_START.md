# 🚀 Jupyter Quick Start Guide

## 1️⃣ Installation (einmalig)

Öffne PowerShell im Projektverzeichnis:

```powershell
cd C:\Users\morri\PycharmProjects\PythonPortfolio
pip install -r requirements.txt
```

Oder installiere das Projekt als Package:

```powershell
pip install -e .
```

## 2️⃣ Jupyter Notebook starten

```powershell
jupyter notebook
```

Dies öffnet Jupyter im Standard-Browser (http://localhost:8888)

## 3️⃣ Notebooks verwenden

### Vorgefertigtes Beispiel-Notebook:
- **`Example_ML_Pipeline.ipynb`** - Komplettes ML-Training aller 8 Modelle mit Vergleichen

### Neue Notebooks erstellen:

1. Klick auf **"New"** → **"Python 3"**
2. Oder: Im Datei-Browser **"New"** → **"Notebook"**

## 4️⃣ Imports in Notebooks

```python
# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Pipeline
from ml.dataset import Dataset
from ml.models.model_factory import ModelFactory
from ml.trainer import Trainer
from ml.metrics.accuracy import Accuracy
from ml.metrics.confusion_matrix import ConfusionMatrix
```

## 5️⃣ Beispiel: Ein Modell trainieren

```python
# Daten laden
dataset = Dataset(test_size=0.3, random_state=42)
dataset.load_data()

# Modell erstellen (aus 8 verfügbaren)
model = ModelFactory.get_model("svm_rbf", input_shape=4, output_shape=3)

# Trainieren
trainer = Trainer(
    model=model, 
    dataset=dataset, 
    metrics_list=[Accuracy()],
    epochs=1
)
trainer.train()

# Evaluieren
results = trainer.evaluate()
print(f"Accuracy: {results['Accuracy']:.4f}")
```

## 📚 Verfügbare Modelle

```python
available_models = [
    "svm_linear",      # SVM mit linearem Kernel
    "svm_rbf",         # SVM mit RBF Kernel
    "svm_poly",        # SVM mit polynomialem Kernel
    "svm_sigmoid",     # SVM mit Sigmoid Kernel
    "decision_tree",   # Decision Tree
    "knn",             # K-Nearest Neighbors
    "random_forest",   # Random Forest
    "lda"              # Linear Discriminant Analysis
]
```

## 🛠️ Debugging & Tipps

### Kernel neustarten
Wenn Imports nicht funktionieren:
- **Kernel** → **Restart**

### Änderungen im Code reloaden
```python
%load_ext autoreload
%autoreload 2
```

### Inline-Plots
```python
%matplotlib inline
```

### Progessbar für lange Operationen
```python
from tqdm import tqdm
for item in tqdm(items):
    # Do something
```

## 📖 Weitere Ressourcen

- **`JUPYTER_README.md`** - Ausführliches Handbuch
- **`launcher.py`** - CLI-Version (Command Line Interface)
- **`Example_ML_Pipeline.ipynb`** - Vollständiges Beispiel mit allen Schritten

## ✅ Häufige Fehler

| Fehler | Lösung |
|--------|--------|
| `ModuleNotFoundError: No module named 'ml'` | `pip install -e .` im Projektverzeichnis ausführen |
| `jupyter: command not found` | `pip install jupyter` ausführen |
| Imports funktionieren nicht | Kernel neustarten: **Kernel** → **Restart** |

---

**Happy Jupyter Coding! 🎉**

