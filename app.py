import os
import joblib
import pandas as pd
import numpy as np

# 1. Definir rutas posibles para el archivo que tienes generado (xgb_model.pkl)
rutas_posibles = [
    '../outputs/xgb_model.pkl',
    'outputs/xgb_model.pkl',
    '../xgb_model.pkl',
    'xgb_model.pkl'
]

MODEL_PATH = None
for ruta in rutas_posibles:
    if os.path.exists(ruta):
        MODEL_PATH = ruta
        break

if MODEL_PATH:
    print(f"✅ Cerebro del modelo encontrado en: {MODEL_PATH}")
else:
    print("❌ No se encontró el archivo 'xgb_model.pkl'.")
    print(f"Rutas intentadas: {rutas_posibles}")

try:
    if not MODEL_PATH:
        raise FileNotFoundError()
        
    # 2. CARGAR EL MODELO DESDE EL ARCHIVO .PKL
    modelo_cargado = joblib.load(MODEL_PATH)
    print("✅ ¡Cerebro cargado con éxito! El modelo está listo para tomar decisiones en tiempo real.\n")

    # 3. BUSCAR LOS SPLITS DE PRUEBA
    ruta_xtest = '../outputs/splits/X_test.pkl' if os.path.exists('../outputs/splits/X_test.pkl') else 'outputs/splits/X_test.pkl'
    ruta_ytest = '../outputs/splits/y_test.pkl' if os.path.exists('../outputs/splits/y_test.pkl') else 'outputs/splits/y_test.pkl'
    
    X_test_loaded = joblib.load(ruta_xtest)
    y_test_loaded = joblib.load(ruta_ytest)

    # Tomemos 2 casos de prueba aleatorios de los guardados en disco
    casos_demo = X_test_loaded.sample(2, random_state=14)
    etiquetas_reales = y_test_loaded.loc[casos_demo.index].values

    print("== EVALUANDO TRANSACCIONES EN TIEMPO REAL ==\n")
    for idx, (i, fila) in enumerate(casos_demo.iterrows()):
        transaccion_entrante = pd.DataFrame([fila])

        # 4. PASARLE LA TRANSACCIÓN AL MODELO (Inferencia en vivo)
        probabilidades = modelo_cargado.predict_proba(transaccion_entrante)[0]
        probabilidad_fraude = probabilidades[1]

        # 5. TOMAR DECISIÓN BASADA EN EL UMBRAL ÓPTIMO (0.74)
        umbral = 0.74
        resultado_modelo = "🚨 RECHAZADA (Sospecha de Fraude)" if probabilidad_fraude >= umbral else "✅ APROBADA (Transacción Segura)"

        print(f"Transacción #{idx+1} (ID: {i}):")
        print(f"  - Monto Original Registrado: ${fila.get('TransactionAmt', 'Desconocido')}")
        print(f"  - Probabilidad de Fraude: {probabilidad_fraude * 100:.2f}%")
        print(f"  - Decisión Automática: {resultado_modelo}")
        print(f"  - Etiqueta Real (La verdad): {'FRAUDE' if etiquetas_reales[idx] == 1 else 'LEGITIMA'}")
        print("-" * 50)

except FileNotFoundError:
    print("\n💡 Si estás ejecutando esto de forma local, asegúrate de colocar el archivo 'xgb_model.pkl' dentro de la carpeta 'outputs' de tu proyecto.")