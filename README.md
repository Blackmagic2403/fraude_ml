```markdown
# 💳 Sistema de Detección de Fraude en Transacciones con Machine Learning

Este repositorio contiene un pipeline completo e industrializado para la detección de fraude en transacciones financieras de comercio electrónico, utilizando el dataset de la competencia **IEEE-CIS Fraud Detection** de Kaggle y el algoritmo **XGBoost**.

El proyecto está completamente preparado para ejecutarse de forma local o de manera aislada y reproducible mediante **Docker**.

---

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu computadora una de las siguientes opciones:

*   **Opción Recomendada:** [Docker Desktop](https://www.docker.com/products/docker-desktop) instalado y ejecutándose.
*   **Opción Local (Alternativa):** [Python 3.10 o superior](https://www.python.org/downloads/) y el gestor de paquetes `pip`.

---

## 🚀 Paso 1: Clonar el Repositorio y Preparar los Datos

1. Abre la terminal de tu sistema operativo (PowerShell o CMD en Windows, Terminal en macOS/Linux) y clona este repositorio:
   ```bash
   git clone https://github.com/Blackmagic2403/fraude_ml.git
   cd fraude_ml
   ```

2. Crea la estructura básica de carpetas requerida en la raíz del proyecto si no existe:
   ```bash
   mkdir data outputs
   ```

3. **Descargar los datos:** Descarga los archivos CSV del dataset oficial en [Kaggle - IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data). Coloca los siguientes archivos dentro de la carpeta `data/`:
    *   `train_transaction.csv`
    *   `train_identity.csv`

La estructura final de tu proyecto debe verse así:
```
mi-proyecto-fraude/
├── data/
│   ├── train_identity.csv
│   └── train_transaction.csv
├── outputs/
├── Dockerfile
├── run_pipeline.py
└── README.md
```

---

## 🐳 Método A: Ejecución con Docker (Recomendado)

Este método no requiere instalar Python ni ninguna librería en tu máquina física; todo se ejecutará dentro de un contenedor aislado con las versiones exactas necesarias.

### 1. Construir la Imagen de Docker
Desde la carpeta raíz del proyecto, ejecuta el siguiente comando para compilar el entorno:
```bash
docker build -t pipeline-fraude .
```

### 2. Ejecutar el Pipeline de Entrenamiento y Generación de Splits
Corre el siguiente comando para iniciar el proceso. Este comando mapea tus carpetas físicas `data/` y `outputs/` dentro del contenedor para leer los datos crudos y escribir el modelo final de vuelta en tu disco duro:

*   **En Windows (PowerShell):**
    ```powershell
    docker run --rm `
      -v "${PWD}/data:/app/data" `
      -v "${PWD}/outputs:/app/outputs" `
      pipeline-fraude python run_pipeline.py
    ```
*   **En macOS / Linux o Git Bash (Windows):**
    ```bash
    docker run --rm \
      -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/outputs:/app/outputs" \
      pipeline-fraude python run_pipeline.py
    ```

Al finalizar la ejecución, verás el archivo `xgb_model.pkl` y los splits de prueba en `outputs/splits/X_test.pkl` y `outputs/splits/y_test.pkl` creados en tu computadora local.

---

## 🐍 Método B: Ejecución Local en tu Máquina (Alternativa)

Si prefieres ejecutar el código directamente sin Docker:

1.  **Crear y activar un entorno virtual:**
    *   *En Windows (PowerShell):*
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   *En macOS / Linux o Git Bash:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
2.  **Instalar dependencias necesarias:**
    ```bash
    pip install numpy pandas scikit-learn xgboost joblib
    ```
3.  **Ejecutar el pipeline:**
    ```bash
    python run_pipeline.py
    ```

---

## 🧪 Paso 3: Simular y Evaluar Decisiones en Tiempo Real

Una vez generado el archivo del modelo `xgb_model.pkl` y los conjuntos de test, puedes probar la inferencia local ejecutando la celda interactiva del Jupyter Notebook o creando un script rápido de prueba en Python (`test_inference.py`) con el siguiente código para evaluar transacciones en milisegundos:

```python
import os
import joblib
import pandas as pd

# Rutas de los artefactos exportados
MODEL_PATH = 'outputs/xgb_model.pkl'
X_TEST_PATH = 'outputs/splits/X_test.pkl'
Y_TEST_PATH = 'outputs/splits/y_test.pkl'

# 1. Cargar el modelo
print("Cargando modelo...")
model = joblib.load(MODEL_PATH)

# 2. Cargar datos de prueba de manera aleatoria
X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)

# Tomar una muestra para la demo
casos = X_test.sample(2, random_state=42)
etiquetas = y_test.loc[casos.index].values

# 3. Realizar predicciones
for idx, (i, fila) in enumerate(casos.iterrows()):
    transaccion = pd.DataFrame([fila])
    prob_fraude = model.predict_proba(transaccion)[0][1]
    
    # Aplicar el umbral óptimo definido del 74%
    resultado = "🚨 BLOQUEAR (Alta probabilidad de Fraude)" if prob_fraude >= 0.74 else "✅ APROBADA"
    
    print(f"Transacción ID {i}:")
    print(f"  - Monto: ${fila.get('TransactionAmt', 0.0):.2f} USD")
    print(f"  - Probabilidad de Fraude: {prob_fraude * 100:.2f}%")
    print(f"  - Decisión del Sistema: {resultado}")
    print(f"  - Valor Real Registrado: {'FRAUDE' if etiquetas[idx] == 1 else 'LEGÍTIMA'}")
    print("-" * 50)
```
```