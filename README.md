# Proyecto Final – Tratamiento de Datos

## 1. Introducción y objetivos

El proyecto final de la asignatura tiene como propósito aplicar las técnicas y conocimientos adquiridos en el curso de **Tratamiento de Datos** para abordar un problema práctico de **aprendizaje automático** basado en datos textuales.

Se trabaja con un conjunto de datos compuesto por publicaciones en redes sociales (tweets) relacionadas con el ámbito político y social, con el objetivo de analizar la difusión de desinformación y su posible relación con la **polarización ideológica** entre los usuarios.

Los principales objetivos del proyecto son:

- Realizar un **análisis exploratorio** de las variables de entrada y visualizar la relación entre variables.
- Diseñar un **pipeline de preprocesamiento y limpieza de texto** (normalización, tokenización, lematización, eliminación de stopwords…).
- Aplicar distintos métodos de **vectorización textual**: TF-IDF, Word2Vec y embeddings contextuales (BERT).
- Entrenar y evaluar modelos de **clasificación supervisada** (red neuronal y algoritmos de Scikit-learn).
- Comparar el rendimiento de las diferentes estrategias de vectorización y modelos, analizando su relación con la detección de desinformación y la polarización ideológica.
- Implementar **fine-tuning** sobre un modelo Transformer preentrenado (Hugging Face) para mejorar la capacidad predictiva en el contexto del corpus político.

A lo largo del documento se detalla el proceso seguido, desde el análisis y preprocesamiento de los datos hasta la implementación y evaluación de los modelos, justificando las decisiones tomadas en cada etapa.

---

## 2. Metodología

Para la realización del trabajo se ha seguido un enfoque eminentemente práctico, fundamentado en los contenidos vistos durante el curso.  

Se ha trabajado principalmente a partir de los **notebooks proporcionados en clase**, entendiéndolos, completándolos y adaptando el código a las necesidades específicas del proyecto:

- Añadiendo nuevas funciones y bloques de código cuando era necesario.
- Ajustando hiperparámetros y configuraciones de modelos.
- Modificando la lógica de preprocesamiento y vectorización para ajustarla al corpus de tweets.

---

## 3. Proyecto básico

El proyecto básico consiste en resolver una tarea de **clasificación multiclase** de ideología política, comparando el rendimiento obtenido al usar diferentes técnicas de vectorización y al menos dos enfoques de aprendizaje automático.

Requisitos abordados:

- Análisis exploratorio de las variables de entrada y de la relación entre las etiquetas y los tweets.
- Representación vectorial de los tweets mediante:
  - **TF-IDF**
  - **Word2Vec**
  - **Embeddings contextuales** (BERT / BETO)
- Entrenamiento y evaluación de modelos de clasificación usando:
  - Una **red neuronal** (PyTorch).
  - Varios clasificadores de **Scikit-learn**.
- Comparación de los resultados anteriores con el **fine-tuning** de un modelo Transformer preentrenado (Hugging Face).

---

## 4. Análisis exploratorio del conjunto de datos

### 4.1 Carga y descripción del conjunto de datos

Se trabaja con el dataset **politicES** (`politicES_phase_2_test_codalab`).  
Mediante la librería **Pandas** se genera un DataFrame con:

- **43.760 instancias**
- **6 variables**

El conjunto combina:

- Texto no estructurado: **`tweet`**
- Metadatos sociodemográficos: **`gender`**, **`profession`**
- Etiquetas de ideología: **`ideology_binary`**, **`ideology_multiclass`**

Se comprobó la integridad con `isnull()`, confirmando **ausencia total de valores nulos**, por lo que no fue necesario aplicar técnicas de imputación.

### 4.2 Distribución por género

Se observa un desbalance importante en la variable **`gender`**:  

- Predominio del género **masculino (`male`)**, con unas 30.480 instancias (~70%).
- Representación minoritaria del género **femenino (`female`)**.

![Distribución por género](/images/iZ9_Image_1.png)

### 4.3 Distribución por profesión

La variable **`profession`** muestra un perfil muy sesgado hacia figuras de relevancia pública:

- La categoría **“journalist”** es claramente dominante (cerca de 25.000 registros).
- Le sigue **“politician”**.
- La categoría **“celebrity”** es residual.

![Distribución por profesión](/images/hvZ_Image_2.png)

Esto sugiere que los tweets tienen un estilo cercano al discurso periodístico/político, alejándose del uso típico de una persona común en redes sociales.

### 4.4 Ideología binaria y multiclase

Para la variable **`ideology_binary`**:

- Mayor presencia de usuarios etiquetados como **“left”** (izquierda).
- Menor frecuencia de **“right”** (derecha).

![Ideología binaria](/images/Hgu_Image_3.png)

Para **`ideology_multiclass`** (cuatro clases):

- Predominan las posturas **moderadas**.
- La clase **“moderate_left”** es la más numerosa, seguida de **“moderate_right”**.
- Las posiciones ideológicas más extremas son menos frecuentes.

![Ideología multiclase](/images/GdD_Image_4.png)

### 4.5 Análisis temático basado en diccionario

Se definieron cinco categorías temáticas principales: **Economía, Salud, Educación, Seguridad y Política**, asociadas a listas de palabras clave.  

El enfoque heurístico clasificó la mayoría de tweets en la categoría **“Otros”** (>35.000 instancias), lo que evidencia:

- Gran **diversidad léxica** del corpus.
- Limitaciones de la clasificación basada solo en **coincidencia de palabras clave**.

![Distribución por tema](/images/PDt_Image_5.png)

### 4.6 Análisis léxico por orientación ideológica

Se dividió el corpus según `ideology_multiclass` en cuatro subconjuntos: `left`, `moderate_left`, `moderate_right`, `right`.  
Sobre cada subconjunto se aplicó el mismo pipeline de preprocesamiento:

- Minúsculas
- Eliminación de URLs, signos de puntuación y números
- Eliminación de tokens poco informativos (`user`, `politician`, `politicalparty`, `hashtag`, etc.)
- Eliminación de **stopwords** en español (NLTK)

Después se tokenizó, se calcularon frecuencias y se generaron **nubes de palabras** con `WordCloud`:

Ideología: Izquierda
![](/images/5ND_Image_6.png)
Ideología: Izquierda moderada
![](/images/oMS_Image_7.png)
Ideología: Derecha moderada
![](/images/geF_Image_8.png)
Ideología: Derecha
![](/images/s7d_Image_9.png)

Este análisis permite comparar el vocabulario predominante de cada bloque ideológico y sirve de base para aplicar técnicas más avanzadas de PLN y aprendizaje automático.

### 4.7 Hipótesis inicial
A priori podemos deducir que la clasificación va a ser compleja por varios aspectos. Tenemos datos desbalanceados lo que va a producir un mal entrenamiento en los clasificadores, aprendiendo de forma muy exacta algunas clases y otras con mayor dificultad. Además, se le suma que las palabras mayoritarias empleadas en cada clase son muy similares, provocando que la diferenciación por este factor no sea muy grande.
Por tanto, nuestra hipótesis inicial es que vamos a obtener unas métricas muy pobres no pudiendo identificar de manera correcta las clases.

---

## 5. Representación vectorial del texto

### 5.1 Preprocesamiento y limpieza

Antes de vectorizar, se construye un pipeline de preprocesamiento para:

- **Mejorar la calidad de los datos** (eliminando ruido, caracteres irrelevantes, etc.).
- **Normalizar y estandarizar el texto** para reducir la variabilidad.

Pasos principales:

1. **Eliminación de filas con `desc` vacío.**  
   Evita introducir ruido y mejora la eficiencia del modelo.
2. **Eliminación de caracteres especiales y números**, manteniendo solo palabras.
3. **Tokenización** de los textos, convirtiendo cada descripción en una lista de tokens.
4. **Homogeneización**, en tres fases:
   - Conversión a **minúsculas**.
   - **Lematización**, para agrupar variantes de una misma palabra.
   - Eliminación de **stopwords**.

A continuación, se muestra con la primera frase del primer tweet el procedimiento de aplicar cada uno de estos pasos.

![](/images/1.png)

![](/images/flecha.png)

![](/images/2.png)
![](/images/flecha.png)
![](/images/3.png)
![](/images/flecha.png)

![](/images/4.png)

### 5.2 Vectorización

Se exploran tres enfoques:

#### 5.2.1 TF-IDF

TF-IDF (**Term Frequency – Inverse Document Frequency**) cuantifica la relevancia de cada término dentro de un documento en relación con el corpus completo:

- Penaliza términos muy frecuentes y poco informativos.
- Potencia palabras específicas y discriminativas.

Pasos:

1. Construcción del **Bag of Words (BoW)**:  
   cada documento se representa como pares `(id_palabra, frecuencia)`.
![](/images/7.png)
   
3. Aplicación de **`TfidfVectorizer`** de Scikit-learn sobre el BoW.
![](/images/8.png)  
4. Conversión final a vectores densos de tamaño |vocabulario| mediante `sparse2full` para poder usarlos en la red neuronal.
![](/images/9.png)  

TF-IDF funciona especialmente bien con clasificadores lineales como **SVM** o **regresión logística**, aunque no captura relaciones semánticas profundas entre palabras.

#### 5.2.2 Word2Vec

**Word2Vec** genera representaciones distribuidas en un espacio vectorial continuo, donde la cercanía entre vectores refleja similitud semántica.

- Entrenado de forma no supervisada con **Gensim** sobre los tweets preprocesados.
- Cada palabra recibe un vector denso.
  ![](/images/F4J_Image_10.png)
- Los documentos se representan como el **promedio** de los vectores de sus palabras.

Ejemplo de exploración semántica con `most_similar()`:

- Para **“mujeres”** aparecen términos como “víctimas”, “violencia”, “género”, “derechos”.
- Para **“gobierno”** aparecen términos como “socialcomunista”, “sedición”, “constitucional”, “oposición”.

![](/images/10.png)

Se proyectaron 500 embeddings con **t-SNE** para visualizar agrupaciones semánticas y comprobar la coherencia del modelo.
![](/images/11.png)

#### 5.2.3 BERT / BETO (embeddings contextuales)

Los embeddings contextuales basados en Transformers (como **BERT**) generan vectores cuyo significado depende del **contexto** en el que aparece cada palabra.

- Se usa el modelo **BETO** (`dccuchile/bert-base-spanish-wwm-cased`), específico para español.
- Por cuestiones computacionales se trabajan solo **2.000 tweets** para la generación de embeddings.

A continuación se muestra la representación BERT del primer tweet.
![](/images/bert1.png)

Tras obtener las representaciones BERT de los tweets, se calculó la **matriz de similitud** (coseno) entre los 5 primeros:

![](/images/bert2.png)
![](/images/bert3.png)

- Todos los valores fuera de la diagonal son > 0.85.
- Algunos pares (por ejemplo, tweet 1 y 3) tienen similitud ~0.96, coherente con su contenido.

Aunque BERT tiene un coste computacional mayor que TF-IDF o Word2Vec, su capacidad para capturar matices lingüísticos lo convierte en una alternativa muy adecuada para tareas de **detección de desinformación** y análisis de **posicionamiento ideológico**.

---

## 6. Modelado y evaluación

El objetivo es clasificar los tweets según su **ideología multiclase**:

- `left`
- `moderate_left`
- `moderate_right`
- `right`

Así,comprobamos hasta qué punto el uso del lenguaje permite diferenciar entre clases.
Para ello vamos a usar una red neuronal y tres clasificadores de la librería Scikit-learn
### 6.1 Red neuronal
Para la red neuronal utilizada se entrena un clasificador a partir de textos usando una de tres representaciones numéricas: TF-IDF, Word2Vec o BERT.
1) Se construyen los vectores de entrada:
- TF-IDF: cada documento se convierte en un vector denso.
- Word2Vec: se cargan los vectores ya generados para cada documento.
- BERT: se cargan los embeddings generados previamente para cada documento.

2) Las etiquetas se convierten a valores numéricos usando LabelEncoder.

3) El programa pide al usuario elegir el tipo de vectorización:
- T = TF-IDF
- W = Word2Vec
- B = BERT

4) Se revisa que los datos no contengan valores NaN.

5) Se entrena un modelo mediante la función train_classifier__ utilizando:
- 300 épocas
- learning rate = 0.0001
- hidden_dim = 256

6) Al finalizar, se muestran gráficas de pérdida, precisión y F1 usando las funciones plot_loss, plot_accuracy y plot_f1.

### 6.2 Clasificadores de Scikit-learn

Se emplean tres algoritmos:

- **Regresión logística**
- **Random Forest**
- **K-NN**

Para cada uno se definen dos funciones:

1. **Función de validación**  
   - Busca los mejores hiperparámetros usando train + val.
2. **Función de clasificación**  
   - Entrena con los hiperparámetros óptimos y evalúa en test.
  
Para ello vamos, se define una función `split_data` que divide los datos en tres subconjutos:

- **70%** entrenamiento (train)
- **20%** validación (val)
- **10%** test
![](/images/8mH_Image_11.png)
En el caso de BERT se ajusta el tamaño de `X` e `Y` para trabajar con un subconjunto de 8.000 muestras.

Las métricas para evaluar los modelos son:

- **Accuracy**
- **F1-score**
- **AUC-ROC**
- **Cross-entropy**
- **Matriz de confusión**

#### Regresión logística

Funciones: `logistic_reg_class` y `logistic_validation`.

- Se estandarizan los datos.
- Se usa un regresor logístico con:
  - Máx. 2000 iteraciones
  - Solver `lbfgs` (apto para multiclase)
  - `class_weight="balanced"` para compensar desbalanceos.

![](/images/bDU_Image_12.png)  
![](/images/ARX_Image_13.png)

#### Random Forest

Funciones: `RandomForest_class` y `randomforest_validation`.

- Hiperparámetros explorados:  
  número de árboles, profundidad máxima, mínimo de muestras por división, mínimo por hoja, número máximo de características…

![](/images/WRc_Image_14.png)  
![](/images/ffB_Image_15.png)

#### K-NN

Funciones: `KNN_class` y `knn_validation`.

- `KNN_class` recibe `K` y los conjuntos de datos, escalando antes las características.
- `knn_validation` recorre distintos valores de `K` hasta `K_max` y selecciona el que mejor resultado obtiene.

![](/images/ryg_Image_16.png)  
![](/images/GUX_Image_17.png)

En total se obtienen **9 combinaciones** (3 vectorizaciones × 3 clasificadores) cuyas métricas se comparan en el informe.
## 6.3 Modelo Transformer preentrenado con *fine-tuning*

### 6.3.1 Preparación de datos

Se seleccionan las columnas:

- `tweet`
- `ideology_multiclass`

Del dataset original (43.760 tweets) se realiza un **submuestreo aleatorio de 8.000 tweets**, fijando `random_state=42` para garantizar la reproducibilidad.

Las etiquetas (`left`, `moderate_left`, `moderate_right`, `right`) se codifican a enteros con `LabelEncoder`, obteniendo una variable `label` con 4 clases (0–3).

![](/images/4zU_Image_18.png)

División del subconjunto:

- Entrenamiento: 5.760 tweets (~72%)
- Validación: 640 tweets (~8%)
- Test: 1.600 tweets (20%)

Al hacer esta división se aseguro de que en los tres grupos hubiera más o menos la misma proporción de cada ideología, para que ninguno quedara con menos de alguna clase, de esta manera, el modelo solo aprende con los datos de entrenamiento.

Después, cada uno de estos tres grupos (train, validación y test) se pasó al formato que usa Hugging Face (Dataset), que es una tabla que solo guarda dos cosas por fila:
el texto del tweet y su label numérico (la ideología codificada como 0, 1, 2 o 3).


![](/images/PPB_Image_19.png)

### 6.3.2 Configuración del Transformer y entrenamiento

Se emplea el modelo **BETO** (`dccuchile/bert-base-spanish-wwm-cased`) cargado con `AutoModelForSequenceClassification`, añadiendo una capa final de clasificación con 4 neuronas.

Tokenización con `AutoTokenizer.from_pretrained`:

- Longitud máxima: 128 tokens
- `padding="max_length"`
- `truncation=True`

Entrenamiento con la clase `Trainer`, usando como métricas **accuracy** y **F1-macro**.

Hiperparámetros principales:

- Learning rate: `2e-5`
- Batch size (train): 16
- Batch size (eval): 32
- Épocas: 3
- Weight decay: 0.01

Durante el entrenamiento, la pérdida desciende desde ≈1.34 hasta ≈0.65, indicando que el modelo está aprendiendo patrones útiles.
Una vez entrenado el modelo Transformer, se utilizó la función evaluate de la clase Trainer para medir su rendimiento. Esta función calcula varias métricas automáticamente; en este trabajo nos fijamos solo en dos:

La accuracy n correctamente y el F1-macro (eval_f1_macro), que resume cómo de bien funciona el modelo teniendo en cuenta las cuatro clases por igual.

![](/images/ni5_Image_20.png)

---
## 7. Evaluación comparativa


Resultados Red Neuronal:
![](/images/redneuronal_accuracy.png)

![](/images/redneuronal_loss.png)

![](/images/redneuronal_f1.png)

Resultados Logistic Regresion:
TF_IDF
![](/images/logistic_tfidf_mconfusion.png)

![](/images/logistic_tfidf_tabla.png)

Word2Vec
![](/images/logistic_word2vec_mconfusion.png)

![](/images/logistic_word2vec_tabla.png)

Bert
![](/images/logistic_bert_mconfusion.png)

![](/images/logistic_bert_tabla.png)

Resultados Random Forest:
TF_IDF
![](/images/randomforest_tfidf_mconfusion.png)

![](/images/randomforest_tfidf_tabla.png)

Word2Vec


Bert

Resultados KNN

TF_IDF

Word2Vec

Bert

Transformer

En el conjunto de validación, el modelo obtiene una accuracy de 0,506 (aproximadamente un 51 % de aciertos) y un F1-macro de 0,485.
En el conjunto de test, que no se ha utilizado en ningún momento durante el entrenamiento, los valores son 0,548 en accuracy (alrededor de un 55 % de aciertos) y 0,518 en F1-macro.

Además, como en este problema hay cuatro posibles ideologías, si el modelo eligiera una clase completamente al azar solo acertaría aproximadamente uno de cada cuatro tweets. El hecho de que el Transformer acierte algo más de la mitad y obtenga un F1-macro cercano a 0,52 muestra que realmente está aprendiendo patrones en el lenguaje de los tweets y es capaz de utilizar esa información para distinguir entre las distintas ideologías

---

##  9.Conclusiones 

En este trabajo se han implementado distintos métodos de vectorización para clasificar tuits según su ideología política. Tras evaluar los resultados obtenidos, se pueden extraer las siguientes conclusiones:

1. BERT:

El uso de BERT requirió recortar los textos debido a limitaciones del modelo, lo que implicó pérdida de información relevante. Esta reducción afectó negativamente a las métricas de evaluación, como se reflejó en el accuracy y en la matriz de confusión.

La limitación en la longitud de los tuits procesados dificultó que el modelo captara el contexto completo, reduciendo así su efectividad.

2. Word2Vec:

La vectorización con embeddings mediante Word2Vec logró mejorar los resultados respecto a BERT. Aun así, el rendimiento obtenido siguió siendo insuficiente para una clasificación sólida, indicando que este método tampoco captó con precisión las sutilezas ideológicas presentes en los tuits.

3. TF-IDF:

El método que presentó el mejor desempeño fue TF-IDF. Su capacidad para resaltar palabras clave permitió obtener métricas superiores frente a los otros modelos. En este caso, la representación basada en frecuencia resultó más adecuada que los embeddings para este conjunto de datos y tarea específica.

En conjunto, los resultados muestran que, para este problema concreto, los métodos más simples y basados en frecuencia pueden ser más eficaces que modelos basados en
embeddings o modelos preentrenados de lenguaje, especialmente cuando los textos son cortos o contienen poco contexto político explícito.

Por otro lado, destacar que implementando el transformer Hugging Face, los resultados obtenidos son similares a los conseguidos mediante los algoritmos de clasificación

Scikit-Learn y el clasificador implementado con PyTorch concluyendo que el tema de clasificación multiclase por ideología no es el idóneo.

Suposiciones

Durante el desarrollo del proyecto se han identificado varias limitaciones que pueden haber influido en los resultados:

1. Presencia de tuits genéricos:

En el conjunto de datos aparecían tuits que no contenían información política suficiente para asignar una ideología. La inclusión de estos mensajes probablemente afectó a la calidad de la clasificación, introduciendo ruido en el aprendizaje de los modelos.

2. Subjetividad en la clasificación política:

La ideología política es un concepto en gran parte subjetivo. Lo que una persona interpreta como un mensaje de izquierdas o derechas puede no coincidir con la percepción de otra. Esta ambigüedad afecta tanto a la etiquetación manual como al rendimiento esperado de los modelos.

3. Posible enfoque alternativo:

Una estrategia más objetiva podría haber sido clasificar los tuits según el sector político al que se refieren (por ejemplo, economía, educación, políticas sociales, etc.) en lugar de intentar inferir la ideología completa. Esto habría reducido la subjetividad y posiblemente mejorado el rendimiento del sistema.

---
##  10. Proyecto de extensión: análisis temático de la desinformación

Se aplica el algoritmo **K-Means** sobre los embeddings de **Word2Vec** para identificar temas recurrentes relacionados con desinformación y discurso polarizado.

- Se trabaja con vectores de **200 dimensiones**.
- Se elige **K = 20** clusters, buscando equilibrio entre detalle temático y coste computacional.

Se obtienen clusters temáticos bien diferenciados (extracto):

```text
Cluster 5:
['españa', 'madrid', 'frente', 'guerra', 'mientras', 'público', 'sanidad', 'pública', 'mayor', 'empleo', 'dinero', 'grandes', 'medidas', 'comunidad', 'trabajadores', 'familias', 'inflación', 'economía', 'salud', 'ayuntamiento']

Cluster 12:
['politician', 'gobierno', 'ley', 'periodista', 'digital', 'congreso', 'poder', 'sólo', 'derecha', 'podemos', 'españoles', 'sino', 'cambio', 'impuestos', 'crisis', 'reforma', 'mayoría', 'cgpj', 'constitución', 'constitucional']

Cluster 17:
['toda', 'persona', 'vida', 'política', 'país', 'mujeres', 'social', 'derechos', 'hombre', 'violencia', 'justicia', 'derecho', 'libertad', 'víctimas', 'sociedad', 'seguridad', 'lucha', 'género', 'instituciones', 'defender']
