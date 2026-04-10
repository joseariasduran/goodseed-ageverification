# Computer Vision for Age Verification: Legal Compliance in Retail
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Resumen Ejecutivo
Este proyecto surge de la necesidad de la cadena de supermercados **Good Seed** de automatizar la verificación de edad en la venta de alcohol para mitigar riesgos legales. Como **Data Scientist**, desarrollé un sistema de visión artificial basado en **Deep Learning** capaz de estimar la edad de los clientes en tiempo real con un margen de error mínimo, optimizando la operación en cajas y garantizando el cumplimiento normativo.

## 🎯 Objetivos del Proyecto
* **Automatización de Cumplimiento:** Detectar proactivamente a menores de edad durante el proceso de pago.
* **Optimización de Procesos:** Reducir la fricción en el punto de venta al automatizar la toma de decisiones basada en datos.
* **Precisión Técnica:** Alcanzar un Error Absoluto Medio (MAE) inferior a 8 años utilizando arquitecturas de vanguardia.

## 🛠️ Stack Tecnológico & Metodología
Dada mi formación técnica en **Python y SQL**, el proyecto se estructuró bajo el siguiente pipeline:

* **Lenguaje:** Python (Pandas, Numpy).
* **Deep Learning Framework:** TensorFlow / Keras.
* **Arquitectura:** **ResNet50** (Transfer Learning) pre-entrenada en ImageNet.
* **Infraestructura:** Procesamiento en GPU para entrenamiento acelerado.
* **Optimización:** Optimizador Adam con tasa de aprendizaje de $0.0001$.

## 📊 Análisis Exploratorio de Datos (EDA)
El conjunto de datos comprende **7,600 imágenes** de rostros con su edad real etiquetada.
* **Distribución:** Se identificó una alta densidad de datos en el rango de adultos jóvenes (20-40 años), crucial para el cumplimiento legal.
* **Pre-procesamiento:** Se implementó `ImageDataGenerator` para normalización de píxeles ($1/255$) y aumento de datos (horizontal flip) para mejorar la robustez del modelo.

## 🧠 Resultados del Modelo
El entrenamiento se realizó durante 10 épocas, logrando una convergencia eficiente:

| Métrica | Valor Final | Meta | Estado |
| :--- | :--- | :--- | :--- |
| **MAE (Test)** | **6.54** | < 8.0 | ✅ Superado |
| **Loss (MSE)** | **74.20** | - | - |

> **Nota Técnica:** El uso de una capa de salida con activación `ReLU` garantiza que el modelo nunca prediga edades negativas, alineándose con la realidad física del problema.

## 💡 Conclusiones y Valor de Negocio
1.  **Fiabilidad:** El modelo es capaz de distinguir con alta precisión entre un menor de edad y un adulto mayor.
2.  **Estrategia de Implementación:** Se recomienda un enfoque de "Filtrado Híbrido": si el modelo predice una edad $< 25$ años, el sistema activa automáticamente una solicitud de identificación física por parte del cajero.
3.  **Escalabilidad:** Al igual que en mis proyectos de **AutoSmart-Retention**, este modelo es escalable a otras áreas como el análisis demográfico para marketing personalizado.

## 📂 Estructura del Proyecto
* `/notebooks`: Jupyter Notebook con el ciclo de vida completo del proyecto.
* `/scripts`: Script `.py` optimizado para ejecución en clusters de GPU.
* `/reports`: Visualizaciones de la distribución de edades y muestras de predicción.

---
**Desarrollado por Jose Arias Duran**
*Ingeniero Industrial | Data Science & Machine Learning Specialist*
[LinkedIn](https://www.linkedin.com/joseariasduran/) | [Portfolio](https://github.com/joseariasduran/)
