# 📋 Command Cheat Sheet

## Installation

```powershell
# Schritt 1: Ins Projektverzeichnis gehen
cd C:\Users\morri\PycharmProjects\PythonPortfolio

# Schritt 2: Dependencies installieren (EINE OPTION)
pip install -r requirements.txt

# ODER das Projekt als Package installieren (EMPFOHLEN)
pip install -e .
```

---

## Jupyter Notebook Starten

```powershell
# Standard
jupyter notebook

# Spezifisches Verzeichnis
jupyter notebook --notebook-dir="C:\Users\morri\PycharmProjects\PythonPortfolio"

# Mit spezifischem Port
jupyter notebook --port=8889

# Im Browser öffnen (falls nicht automatisch)
# http://localhost:8888
```

---

## Launcher.py (CLI Version)

```powershell
python launcher.py
```

Interaktives Menü zum:
- Modell auswählen
- Hyperparameter konfigurieren
- Training durchführen
- Ergebnisse anschauen
- Weitere Runden durchführen

---

## Direktes Python Scripting

```python
# Imports
from ml.dataset import Dataset
from ml.models.model_factory import ModelFactory
from ml.trainer import Trainer
from ml.metrics.accuracy import Accuracy
from ml.metrics.confusion_matrix import ConfusionMatrix

# Ein vollständiges Beispiel
dataset = Dataset(test_size=0.3, random_state=42)
dataset.load_data()

model = ModelFactory.get_model("svm_rbf", input_shape=4, output_shape=3)
trainer = Trainer(model=model, dataset=dataset, 
                  metrics_list=[Accuracy(), ConfusionMatrix()], epochs=1)
trainer.train()
results = trainer.evaluate()

for name, value in results.items():
    print(f"{name}:\n{value}\n")
```

---

## Jupyter Notebook Magic Commands

```python
# Im Notebook am Anfang eingeben:

# Auto-reload von Modulen aktivieren
%load_ext autoreload
%autoreload 2

# Inline-Plots anzeigen
%matplotlib inline

# Größe der Plots einstellen
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

# Code-Laufzeit messen
%timeit function_name()

# Memory-Nutzung anzeigen
%memory

# System-Befehle ausführen
!pip list
!dir
```

---

## Package Installation Optionen

```powershell
# Editable Mode (Entwicklung - EMPFOHLEN)
pip install -e .

# Mit zusätzlichen Features
pip install -e ".[dev]"  # Falls extras definiert

# Nur Update
pip install --upgrade -e .

# Neuerstellung
pip install --force-reinstall -e .
```

---

## Testing & Debugging

```powershell
# Python-Syntax prüfen
python -m py_compile ml/models/*.py

# Imports testen
python -c "from ml.dataset import Dataset; print('OK')"

# Alle Modelle testen
python -c "
from ml.models.model_factory import ModelFactory
models = ['svm_linear', 'svm_rbf', 'svm_poly', 'svm_sigmoid', 'decision_tree', 'knn', 'random_forest', 'lda']
for m in models:
    try:
        ModelFactory.get_model(m, 4, 3)
        print(f'✓ {m}')
    except Exception as e:
        print(f'✗ {m}: {e}')
"
```

---

## Jupyter-Kernel verwalten

```powershell
# Liste alle verfügbaren Kernels
jupyter kernelspec list

# Python Kernel installieren
python -m ipykernel install --user --name py311 --display-name "Python 3.11"

# Kernel entfernen
jupyter kernelspec remove py311
```

---

## Notebooks als Python-Scripts konvertieren

```powershell
# Notebook zu Python konvertieren
jupyter nbconvert --to python Example_ML_Pipeline.ipynb

# Output wird: Example_ML_Pipeline.py
```

---

## Jupyter Konfiguration

```powershell
# Konfig-Verzeichnis anzeigen
jupyter --config-dir

# Konfiguration generieren
jupyter notebook --generate-config

# Token anzeigen (falls Authentifizierung nötig)
jupyter notebook list
```

---

## Häufige Probleme & Lösungen

```powershell
# Problem: ModuleNotFoundError: No module named 'ml'
# Lösung:
pip install -e .
# ODER: Starte Python aus dem Projektverzeichnis

# Problem: Jupyter findet meinen Kernel nicht
# Lösung:
python -m ipykernel install --user

# Problem: Port 8888 ist bereits in Benutzung
# Lösung:
jupyter notebook --port=8889

# Problem: Änderungen im Code werden nicht übernommen
# Lösung: Kernel neustarten
# Menü: Kernel → Restart
# ODER im Code:
%load_ext autoreload
%autoreload 2
```

---

## Performance & Optimierung

```python
# Im Notebook: Schnellerer Training möglich mit:
import warnings
warnings.filterwarnings('ignore')

# Multi-Processing für Random Forest
model = ModelFactory.get_model("random_forest", 4, 3)
# Bereits unterstützt durch sklearn

# Echtzeit-Monitoring
from tqdm import tqdm
for epoch in tqdm(range(10)):
    # Training code
```

---

## Dokumentation & Hilfe

```python
# Im Notebook:

# Funktion-Hilfe anzeigen
help(ModelFactory.get_model)

# Oder
?ModelFactory.get_model

# Code-Quelle anzeigen
??ModelFactory.get_model

# Attribute/Methoden auflisten
ModelFactory.<TAB>  # Drücke Tab für Auto-Completion
```

---

## Dateistruktur schnell navigieren

```powershell
# Im Projektverzeichnis
tree /F                    # Alle Dateien anzeigen
dir /S *.py               # Alle Python-Dateien

# Mit Details
Get-ChildItem -Recurse -Include "*.py" | Select-Object FullName
```

---

**Viel Spaß beim Coden! 🚀**

