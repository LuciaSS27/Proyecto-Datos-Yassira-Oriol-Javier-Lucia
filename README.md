# 

**Proyecto Final prueba**

**Tratamiento de Datos**

# Introducci?n y objetivos

El proyecto final de la asignatura tiene como prop?sito aplicar las t?cnicas y conocimientos adquiridos en el curso de Tratamiento de Datos para abordar un problema pr?ctico de aprendizaje autom?tico basado en datos textuales.

En este caso, se trabajar? con un conjunto de datos compuesto por publicaciones en redes sociales (tweets) relacionadas con el ?mbito pol?tico y social, con el objetivo de analizar la difusi?n de desinformaci?n y su posible relaci?n con la polarizaci?n ideol?gica entre los usuarios. Los objetivos para este proyecto son:

- An?lisis de las variables de entrada y visualizaci?n de la relaci?n entre variables.

- Preprocesamiento y limpieza del texto, mediante la implementaci?n de un pipeline eficiente para la normalizaci?n y tokenizaci?n de los mensajes.

- Vectorizaci?n textual aplicando distintos m?todos de representaci?n: TF?IDF, Word2Vec y embeddings contextuales (BERT).

- Entrenamiento y evaluaci?n de modelos de clasificaci?n utilizando redes neuronales y otros algoritmos de aprendizaje supervisado.

- Comparaci?n del rendimiento entre las diferentes estrategias de vectorizaci?n y modelos, analizando su relaci?n con la detecci?n de desinformaci?n y la polarizaci?n ideol?gica.

- Implementaci?n de fine?tuning sobre un modelo Transformer preentrenado (Hugging?Face) para mejorar la capacidad predictiva en el contexto del corpus pol?tico.

A lo largo de este documento se detallar? el proceso llevado a cabo para alcanzar los objetivos planteados, abarcando desde el an?lisis y preprocesamiento de los datos hasta la implementaci?n y evaluaci?n de modelos de aprendizaje autom?tico. Se describir?n las t?cnicas utilizadas para la vectorizaci?n textual y los enfoques empleados para resolver la tarea de clasificaci?n, justificando las decisiones tomadas en cada etapa. Adem?s, se analizar?n los resultados obtenidos mediante m?tricas adecuadas, comparando el rendimiento de las diferentes metodolog?as aplicadas.

# Metodolog?a

Para la realizaci?n de este trabajo, se ha seguido un enfoque pr?ctico fundamentado en

los conocimientos adquiridos durante el curso, adaptando y aplicando las t?cnicas vistas

en clase para abordar las distintas etapas del proyecto. Se ha trabajado principalmente

mediante los notebooks vistos en clase, entendi?ndolos, complet?ndolos y ajustando el

c?digo e implementaciones a las necesidades espec?ficas de este proyecto.

# Proyecto b?sico

El proyecto b?sico consiste en resolver una tarea de clasificaci?n, comparando el rendimiento obtenido al usar diferentes t?cnicas de vectorizaci?n de los documentos y al menos dos enfoques distintos de aprendizaje autom?tico. Se detallan los requisitos espec?ficos a continuaci?n:

- An?lisis exploratorio de las variables de entrada: visualizaci?n de la relaci?n entre la variable de salida y los tweets.

- Representaci?n vectorial de los tweets mediante tres procedimientos diferentes: TF?IDF, Word2Vec y Embeddings contextuales.

- Entrenamiento y evaluaci?n de modelos de clasificaci?n usando PyTorch y un modelo de clasificaci?n de la librer?a Scikit?learn.

- Comparaci?n de lo obtenido en el paso 3 con el fine-tunning de un modelo preentrenado con Hugging Face.

# An?lisis de las variables de entrada

**1.1. Carga y Descripci?n del Conjunto de Datos**

La fase inicial del proyecto se basa en la carga y el examen estad?stico del conjunto de datos politicES(archivo politicES_phase_2_test_codalab), paso cr?tico para comprender la distribuci?n de las variables y detectar posibles sesgos antes del modelado.

Mediante el uso de la librer?a Pandas, se gener? un DataFrame con una dimensionalidad de 43.760 instancias y 6 variables. Este conjunto combina informaci?n textual no estructurada **(tweet) **con metadatos sociodemogr?ficos clave **(gender, profession)** y etiquetas de posicionamiento ideol?gico **(ideology_binary, ideology_multiclass).**

Finalmente, se someti? el dataset a una verificaci?n de integridad mediante la funci?n `isnull()`. El an?lisis confirm? la ausencia total de valores nulos, garantizando la completitud de la muestra y eliminando la necesidad de aplicar t?cnicas de imputaci?n preliminares."

Una vez cargados los datos, es fundamental realizar un an?lisis descriptivo de las variables categ?ricas disponibles para comprender el perfil sociodemogr?fico y pol?tico de la muestra. A continuaci?n, se examinar? la distribuci?n de cada una de las etiquetas (gender, profession, ideology..) de forma individualizada. El objetivo de este an?lisis es identificar las clases predominantes y detectar posibles desbalanceos en los datos que pudieran condicionar el comportamiento de los modelos predictivos posteriores

Al analizar la composici?n demogr?fica de la muestra en funci?n de la variable **gender**, se identifica un desbalanceo significativo entre las clases. Como se aprecia en la figura, existe una clara predominancia del g?nero masculino (male), el cual cuenta con 30.480 instancias, representando aproximadamente el 70% del total de los datos, frente a una representaci?n minoritaria del g?nero femenino (female).

![](/images/iZ9_Image_1.png)

Posteriormente, se analiz? el perfil profesional de los usuarios mediante la variable **profession**. Como se observa en el gr?fico de barras, la muestra no es representativa de la poblaci?n general, sino que est? altamente sesgada hacia figuras de relevancia p?blica. La categor?a predominante es "Periodista", que abarca la mayor parte de los registros (cerca de 25.000), seguida por la clase "Pol?tico". La categor?a "Celebridad" tiene una presencia residual. Esto sugiere que los tweets tendr?n un estilo profesional y period?stico, diferente a c?mo escribir?a una persona com?n en una red social.

![](/images/hvZ_Image_2.png)

Para la caracterizaci?n pol?tica de la muestra, se examin? en primer lugar la variable **ideology_binary,** que divide a los usuarios en dos grandes bloques pol?ticos. El an?lisis de frecuencias revela una asimetr?a en la distribuci?n de las clases: existe una mayor representaci?n de usuarios etiquetados como **'Izquierda'**, superando las 25.000 instancias, en comparaci?n con la etiqueta **'Derecha',** que cuenta con una presencia menor (aproximadamente 17.500 instancias). Este sesgo hacia la izquierda deber? ser tenido en cuenta durante la fase de entrenamiento del modelo.

![](/images/Hgu_Image_3.png)

Con el objetivo de obtener una mayor exactitud en el perfilado pol?tico, se analiz? la variable **ideology_multiclass. **Al desglosar el espectro pol?tico en cuatro categor?as, se evidencia que las posturas moderadas son predominantes en el conjunto de datos.

Como muestra la figura, la clase **'Izquierda Moderada'** es la m?s numerosa, seguida de la **'Derecha Moderada'**. Por el contrario, las posiciones ideol?gicas m?s definidas o extremas presentan una frecuencia menor. Esto sugiere que, aunque existe un sesgo hacia la izquierda, la muestra se concentra principalmente en el centro del tablero pol?tico.

![](/images/GdD_Image_4.png)

Finalmente, se realiz? una aproximaci?n inicial al contenido de los mensajes mediante una clasificaci?n basada en diccionario. Se definieron cinco categor?as tem?ticas principales (Econom?a, Salud, Educaci?n, Seguridad y Pol?tica) asociadas a listas de palabras clave espec?ficas.

Como se evidencia en la gr?fica, el m?todo heur?stico clasific? la gran mayor?a de las instancias (m?s de 35.000 tweets) en la categor?a **"Otros"**. Este resultado pone de manifiesto la diversidad l?xica del corpus y las limitaciones de los m?todos de clasificaci?n basados estrictamente en coincidencia de palabras clave. Las categor?as espec?ficas como "Pol?tica" o "Econom?a" capturaron una fracci?n menor de los datos, lo que justifica la necesidad de aplicar t?cnicas m?s avanzadas de Procesamiento de Lenguaje Natural (NLP) en etapas subsiguientes.

![](/images/PDt_Image_5.png)

Adem?s del an?lisis tem?tico basado en diccionarios, se llev? a cabo un an?lisis l?xico por orientaci?n ideol?gica utilizando la variable ideology_multiclass del conjunto de datos. Para ello, se dividi? el corpus en cuatro subconjuntos correspondientes a las categor?as *left*, *moderate_left*, *moderate_right* y *right*. Sobre cada uno de estos subconjuntos se aplic? el mismo proceso de preprocesamiento de texto: conversi?n a min?sculas, eliminaci?n de URLs, signos de puntuaci?n y n?meros, supresi?n de tokens poco informativos propios del dataset (como menciones gen?ricas a user, politician, politicalparty o hashtag) y filtrado de *stopwords* en espa?ol utilizando la librer?a NLTK.

Una vez limpiado el texto, se concatenaron los tweets de cada grupo ideol?gico y se tokenizaron para obtener las palabras individuales. Posteriormente, se calcul? la frecuencia de aparici?n de cada t?rmino y se seleccionaron las palabras m?s frecuentes de cada subconjunto. A partir de estas frecuencias se generaron nubes de palabras utilizando la librer?a WordCloud de Python, de modo que las palabras con mayor frecuencia aparecen visualmente con un tama?o m?s grande.

![](/images/5ND_Image_6.png)

![](/images/oMS_Image_7.png)

![](/images/geF_Image_8.png)

![](/images/s7d_Image_9.png)

Este procedimiento permite comparar de forma sistem?tica el vocabulario predominante en cada bloque ideol?gico y proporciona una primera aproximaci?n a las diferencias en el lenguaje empleado por los distintos grupos. Dichas diferencias l?xicas complementan los resultados de la clasificaci?n heur?stica por temas y sirven como base para el uso posterior de t?cnicas m?s avanzadas de Procesamiento de Lenguaje Natural y modelos de aprendizaje autom?tico.

# Representaci?n vectorial del texto

## <span style="text - decoration: underline;">5.1 Preprocesamiento y limpieza de texto</span>

Para poder realizar las vectorizaciones y representaciones, as? como el entrenamiento y la evaluaci?n de los modelos desarrollados, se ha llevado a cabo previamente un proceso de preprocesamiento de texto.

Esta fase resulta esencial para mejorar la calidad de los datos y garantizar que el contenido textual sea adecuado para el an?lisis posterior. Esta fase de preprocesamiento aportar? en los datos:

- Mejora en la calidad de los datos gracias a la limpieza y a la estructuraci?n que

se hace de estos. Se eliminan caracteres innecesarios, palabras irrelevantes

(como las stop words), y otros elementos que no aportan informaci?n significativa

para el modelo.

- Normalizaci?n y estandarizaci?n del texto para reducir la variabilidad de los

datos.

El procedimiento que se ha seguido para el preprocesamiento de los datos es el

siguiente:

1. Eliminaci?n de las filas que en el campo ?desc? tengan textos vac?os. Los textos

vac?os no aportan informaci?n relevante al modelo, lo que podr?a afectar

negativamente el rendimiento del an?lisis. Adem?s, mantener estas filas puede

introducir ruido en el modelo, distorsionando los resultados. Por lo tanto, eliminar

las filas vac?as mejora la precisi?n y eficiencia del preprocesamiento.

2. Eliminaci?n de caracteres especiales y n?meros de los textos de cada

descripci?n. Se mantiene las palabras. De esta manera simplificamos el texto al

m?ximo posible.

3. Una vez eliminados los caracteres especiales y los n?meros, se han tokenizado

los textos. Este paso es crucial porque convierte el texto en una estructura que

los algoritmos pueden procesar f?cilmente. Los principales beneficios de la

tokenizaci?n son la simplificaci?n del texto y la reducci?n de complejidad de este.

4. Despu?s de realizar una nueva estructura en el texto mediante el proceso de

tokenizar, se ha realizado la homogeneizaci?n para estandarizar y limpiar los

datos. Este proceso se ha dividido en tres fases:

- Pasar a min?sculas para eliminar la variabilidad innecesaria del texto,

mejorando la consistencia y reduciendo la complejidad, asegurando que

las palabras se reconozcan sin importar c?mo se escriban.

- Lemmataizer: Con esta t?cnica se reduce las palabras a su ra?z para as?

poder agrupar variantes de una palabra bajo un mismo t?rmino.

- Eliminar las stopwords que no aportan valor sem?ntico y que pueden

distorsionar el an?lisis.

A continuaci?n, se muestra con la primera frase del primer tweet el procedimiento de aplicar cada uno de estos pasos.

## **5.2 Vectorizaci?n**

La vectorizaci?n es un procedimiento clave a la hora de tratar datos ya que convierte el

texto, en este caso el preprocesado, en una forma que puede ser entendida y procesada

por los modelos de aprendizaje autom?tico. En este apartado, se han explorado tres enfoques diferentes para la representaci?n vectorial de los documentos:

- TF-IDF (Term Frequency-Inverse Document Frequency): Este enfoque mide la

importancia de las palabras en un documento en relaci?n con su frecuencia en

el corpus completo.

- Word2Vec: En este enfoque, los documentos se representan como el promedio

de los embeddings de las palabras que los componen, lo que captura la similitud

sem?ntica entre palabras en funci?n de su contexto en el corpus.

- Embeddings contextuales calculados a partir de modelos basados en

transformers: Modelos como BERT y RoBERTa utilizan redes neuronales de

transformers para generar representaciones contextuales de las palabras, es

decir, palabras que pueden tener diferentes significados seg?n el contexto.

Antes de realizar un an?lisis detallado de cada enfoque, es importante destacar la

creaci?n de un diccionario o corpus a partir del texto preprocesado utilizando la librer?a

Gensim. Este diccionario contiene todas las palabras presentes en el texto

preprocesadas, las cuales est?n ordenadas alfab?ticamente. A continuaci?n, se

presenta el resultado del diccionario generado.

Cabe remarcar que este diccionario como indica la imagen anterior contiene 54747

t?rminos por lo que ha sido sometido a un filtrado donde se han conservado las palabras

que aparezcan en al menos 30 tweets del corpus y se han eliminado las palabras

que aparezcan en m?s del 75% de los documentos del corpus.

**TF-IDF**

TF-IDF (Term Frequency - Inverse Document Frequency) es una t?cnica cl?sica de representaci?n vectorial utilizada en Procesamiento de Lenguaje Natural que permite cuantificar la relevancia de cada t?rmino dentro de un documento en relaci?n con el corpus completo. A diferencia de modelos basados ?nicamente en la frecuencia de palabras, TF-IDF penaliza aquellos t?rminos que aparecen de forma recurrente en la mayor?a de los documentos (como stopwords o expresiones comunes), y otorga mayor peso a palabras espec?ficas que pueden resultar discriminativas para la tarea de clasificaci?n.

Antes de aplicar la vectorizaci?n, se ha realizado un paso previo, la creaci?n del BoW (Bag of Words), d?nde  cada token esta representado con (id_palabra, frecuencia ). Es decir, un id ?nico para cada token, y la frecuencia del token dentro de cada tweet. Como en ejemplos anteriores, a continuaci?n se muestra la representaci?n BoW del primer tweet.

El siguiente paso, ha sido aplicar TF-IDF al BoW generado. Para llevar a cabo este proceso, se ha aplicado la implementaci?n de TfidfVectorizer de la librer?a Scikit-learn.

El resultado de este, es una representaci?n parecida a la del BoW, per? esta vez, envez de la frecuencia, con su valor TF-IDF del token.

Finalmente, para poder trabajar con TF-IDF en la red neuronal, se ha convertido cada representaci?n TF?IDF en un vector denso del tama?o del vocabulario con sparse2full.

TF-IDF es especialmente ?til para clasificadores lineales como SVM o regresi?n log?stica, ya que permite capturar patrones de frecuencia asociados con la ideolog?a expresada en el texto.

No obstante, TF-IDF presenta limitaciones inherentes, como su incapacidad para capturar relaciones sem?nticas entre palabras o su sensibilidad al vocabulario exacto utilizado en el conjunto de entrenamiento, lo que dificulta su generalizaci?n ante variaciones ling??sticas. Por este motivo, se complementa su uso con modelos m?s avanzados basados en embeddings distribucionales.

**Word2Vec**

Word2Vec es una t?cnica de representaci?n distribuida basada en redes neuronales que permite proyectar palabras en un espacio vectorial continuo, donde la proximidad entre vectores refleja similitudes sem?nticas aprendidas a partir del contexto en el que aparecen los t?rminos. A diferencia de TF-IDF, el enfoque de Word2Vec no depende de la frecuencia aislada de las palabras, sino de su co-ocurrencia contextual a lo largo del corpus.

Para este proyecto se ha utilizado la librer?a Gensim para entrenar el modelo Word2Vec de forma no supervisada sobre el conjunto de tweets preprocesados.

El modelo resultante genera un vector denso para cada t?rmino del diccionario, permitiendo representar palabras sem?nticamente similares con vectores cercanos en el espacio latente.

![](/images/F4J_Image_10.png)

Para explorar la relaci?n sem?ntica entre palabras se ha usado la funci?n most_similar() que permite calcular qu? palabras est?n m?s cercanas a un t?rmino dado dentro del espacio vectorial, bas?ndose en la similitud del coseno entre sus embeddings.

Para ello, se ha usado palabras actuales en el d?a a d?a de la pol?tica mundial, ?mujeres? y ?gobierno?.

El modelo muestra, por ejemplo, que la palabra "mujeres" se asocia con t?rminos como "v?ctimas", "violencia", "g?nero" o "derechos", lo que indica que ha captado correctamente el contexto social y tem?tico en el que suele aparecer.

En el caso de "gobierno", las palabras m?s cercanas son "socialcomunista", "sedici?n", "constitucional" o "oposici?n", lo que refleja un entorno pol?tico y discursivo coherente con el contenido del corpus.

Estas relaciones confirman que el modelo Word2Vec ha aprendido representaciones sem?nticas ?tiles, donde la proximidad entre vectores refleja similitud de significado o de contexto.

Finalmente, se han representado 500 embeddings utilizando la visualizaci?n T-SNE para analizar las relaciones sem?nticas que captura el modelo. A continuaci?n, se incluyen algunas capturas destacadas que evidencian que el modelo ha aprendido correctamente.

**BERT**

Los embeddings contextuales basados en arquitecturas Transformers representan un avance significativo en la vectorizaci?n de texto, ya que permiten generar representaciones din?micas donde el significado de cada palabra depende expl?citamente del contexto en el que aparece. En lugar de asignar un vector fijo por t?rmino, modelos como BERT (Bidirectional Encoder Representations from Transformers) procesan el texto mediante mecanismos de atenci?n bidireccional, capturando dependencias sem?nticas complejas a nivel de frase.

Cabe mencionar que, para reducir el coste computacional y disminuir el tiempo de ejecuci?n, trabajaremos ?nicamente con 2000 tweets.

A continuaci?n se muestra la representaci?n BERT del primer tweet.

Una vez se ha conseguido obtener la representaci?n BERT para los 2000 tweets, se ha analizado si es que el modelo est? extrayendo conclusiones correctas, es decir evaluaremos si est? analizando bien. Para ello se han usado los 5 primeros tweets y se ha calculado su similitud.

Como se puede observar en la matriz de similitud entre los 5 primeros *tweets, *todos los valores fuera de la diagonal est?n por encima de 0.85, y muchos rondan 0.95.Esto nos indica que los 5 tweets son muy similares en contenido o tem?tica seg?n BERT.

Lo que m?s destaca es que los tweets 1 y 3 tienen una similitud de 0.963, casi id?nticos. Dicha interpretaci?n es correcta ya que el tono y el tema del discurso son muy similares.

Para concluir, aunque BERT ofrece un rendimiento superior en la mayor?a de tareas, su mayor coste computacional y el tiempo de entrenamiento lo convierten en una opci?n m?s exigente respecto a TF-IDF y Word2Vec. Aun as?, su capacidad para representar matices ling??sticos lo convierte en la alternativa m?s adecuada en escenarios de detecci?n de desinformaci?n y an?lisis de posicionamiento ideol?gico.


**6. Modelado y evaluaci?n**

En esta secci?n se emplear?n los datos vectorizados para clasificar los tweets por ideolog?a multiclase, es decir clasificar en:

- Left

- Moderate-left

- Moderate-rigth

- Rigth

As?, podremos comprobar si emplear determinadas palabras o no permite al algoritmo poder diferenciar entre una clase y otra.

## **6.1 Red neuronal**

## **6.2 Algoritmos de Scikit-Learn**

Se ha decidido implementar tres algoritmos de la libreria de Scikit-Learn:

- Logistic Regression

- Random Forest

- K-NN

El primer paso es dividir los datos en tres subconjuntos:

- Entrenamiento / train : datos empleados para entrenar el modelo

- Validaci?n / val : datos empleados para la obtenci?n de los hyperpar?metros

- Test: datos empleados para el an?lisis del clasificador.

Para ello se ha creado una funci?n llamada *split_data *que divide los datos en:

- 70% train

- 20% val

- 10% test

![](/images/8mH_Image_11.png)

Adicionalmente, se ha ajustado el tama?o de X e Y para gestionar el caso del uso de la vectorizaci?n BERT ya que hemos reducido las muestras a 8000.

Posteriormente, se comienzan a realizar los trabajos con cada clasificador. Para ello se han creado dos funciones:

- <span style="text - decoration: underline;">Funci?n de validaci?n</span>: donde usaremos los datos de Train y Val para obtener los hiperpar?metros m?s ?ptimos 

- <span style="text - decoration: underline;">Funci?n de clasificaci?n</span>: usamos los hiperpar?metros obtenidos en la funci?n previa y obtenemos diferentes m?tricas para analizar el rendimiento. Las m?tricas que vamos a evaluar son: accuracy, f1-score, AUC-ROC, cross-entropy y confusi?n matrix.

A continuaci?n, se explica de forma breve los detalles de los procedimientos espec?ficos para cada clasificador.

**Logistic regressor**

Se han implementado las funciones de* logistic_reg_class* y de *logistic_validation*.

La primera de ellas implementa el clasificador, donde introduciremos como par?metros el valor de C y los conjuntos de datos.

![](/images/bDU_Image_12.png)

Dentro de la funci?n se estandarizan los datos y se realiza la propia clasificaci?n. Se ha empleado un logistic regresor con un n?mero m?ximo de iteraciones de 2000, el solver ?lbfgs? ?ptimo para clasificaci?n multiclase y un class_weight balanced para solventar el problema de desbalanceo de clases.

Con el objetivo de mejorar la precisi?n del clasificaci?n, previamente a entrenarlo emplearemos la funci?n de validaci?n d?nde vamos a evaluar diferentes tipos de valores de C.

![](/images/ARX_Image_13.png)

Esta funci?n se encarga de probar cada uno de los valores de C contenidos en el array llaamdo C_values y obtener el que menor error de validaci?n proporciona el cual usaremos para entrenar a nuestro modelo.

**Random Forest**

Siguiendo el ejemplo del anterior modelo, se han implementado dos funciones: *RandomForest_class* y* randomforest_validation*.

En este caso la funci?n del clasificador va a tener como par?metros de entrada los conjuntos de datos, n?mero de ?rboles,profundidad m?xima, n?mero m?nimo de divisi?n,m?nimo de datos por hoja y m?ximo de caracter?sticas por split.

![](/images/WRc_Image_14.png)

La funci?n de validaci?n evalua diferentes conjuntos de valores de estos par?metros definidos en la lista de diccionarios param_grid retornando los mejores valores para nuestro modelo.

![](/images/ffB_Image_15.png)

**KNN**

Finalmente, el ?ltimo clasificador continua con la misma l?gica que los anteriores.

Definimos dos funciones:  *KNN_class* y *knn_validation.*

La funci?n de clasificaci?n *KNN_class* recibe como par?metro el valor de K y los conjuntos de datos. Dentro de la funci?n se escalan los datos para el correcto funcionamiento del algoritmo.

![](/images/ryg_Image_16.png)

Por ?ltimo definimos la funci?n *knn_validation, * que iterar? diferentes valores de K definiendo el valor de K_max y devolver? el mejor valor que ser? empleado en el clasificador.

![](/images/GUX_Image_17.png)

**Empleo de las funciones**

Una vez definidas las funciones a emplear se ejecutar?n cada una de ellas por cada vectorizaci?n de los datos, obteniendo en total 9 ejecuciones con sus respectivas m?tricas que ser?n evaluadas en el siguiente cap?tulo del informe.

**Modelo Transformer preentrenado con *****fine-tuning***

 <span style="text - decoration: underline;">Preparaci?n de datos</span>

 \
Se seleccionaron las columnas tweet  e ideology_multiclass (etiqueta de ideolog?a) del dataset original, que contiene 43.760 tweets. Con el objetivo de reducir el tiempo de entrenamiento del modelo BERT, se realiz? un submuestreo aleatorio de 8000 tweets, fijando una semilla (random_state= 42) para garantizar la reproducibilidad del experimento.

Las etiquetas categ?ricas de ideolog?a (left, moderate_left, moderate_right, right) se codificaron a valores enteros mediante LabelEncoder, obteni?ndose una nueva variable label con 4 clases (0?3), que es la que se utiliza como salida del modelo de clasificaci?n.

![](/images/4zU_Image_18.png)

A continuaci?n, este subconjunto se dividi? en tres particiones mutuamente excluyentes:

- Entrenamiento: 5760 tweets (?72 %)

- Validaci?n: 640 tweets (?8 %)

- Test: 1600 tweets (20 %)

Al hacer esta divisi?n se aseguro de que en los tres grupos hubiera m?s o menos la misma proporci?n de cada ideolog?a, para que ninguno quedara con menos de alguna clase, de esta manera, el modelo solo aprende con los datos de entrenamiento.

Despu?s, cada uno de estos tres grupos (train, validaci?n y test) se pas? al formato que usa Hugging Face (Dataset), que es una tabla que solo guarda dos cosas por fila: \
el texto del tweet y su label num?rico (la ideolog?a codificada como 0, 1, 2 o 3).

![](/images/PPB_Image_19.png)

 <span style="text - decoration: underline;">Configuraci?n del Transformer y </span>*<span style="text - decoration: underline;">fine-tuning*</span>

Para la parte de Transformers se emple? el modelo BETO (dccuchile/bert-base-spanish-wwm-cased), un modelo BERT preentrenado en grandes corpus de texto en espa?ol. El modelo se carg? mediante AutoModelForSequenceClassification, a?adiendo una capa de clasificaci?n con 4 neuronas en la salida, correspondientes a las cuatro ideolog?as consideradas.

El texto de los tweets se proces? con el tokenizer asociado al mismo modelo (AutoTokenizer.from_pretrained), utilizando las siguientes opciones:

- longitud m?xima de secuencia: 128 tokens,

- padding a longitud fija (padding="max_length"),

- truncado de secuencias demasiado largas (truncation=True).

De este modo, cada tweet queda representado por un vector de identificadores de tokens y una m?scara de atenci?n.

El entrenamiento se llev? a cabo con la clase Trainer de Hugging Face, especificando como m?trica de evaluaci?n la accuracy y el F1-macro. Estos valores se calculan a partir de las predicciones del modelo mediante una funci?n compute_metrics que aplica argmax sobre los logits y posteriormente llama a accuracy_score y f1_score(average="macro").

Los principales hiperpar?metros de entrenamiento fueron:

- tasa de aprendizaje (learning rate): 2?10??,

- batch size de entrenamiento: 16,

- batch size de evaluaci?n: 32,

- n?mero de ?pocas: 3,

- weight decay: 0,01.

Durante el entrenamiento la funci?n de p?rdida de entrenamiento disminuye desde valores iniciales en torno a 1,34 hasta aproximadamente 0,65 al final de la tercera ?poca, lo que indica que el modelo est? aprendiendo patrones ?tiles a partir del texto.

![](/images/ni5_Image_20.png)

<span style="text - decoration: underline;">Resultados del Transformer</span>

Tras el fine-tuning, el modelo se evalu? primero sobre el conjunto de validaci?n y posteriormente sobre el conjunto de test, que se ha mantenido completamente independiente durante todo el proceso. Las m?tricas obtenidas fueron:

- Validaci?n

- accuracy = 0,506

- F1-macro = 0,485

- Test

- accuracy = 0,548

- F1-macro = 0,518

Dado que el problema tiene cuatro clases aproximadamente equilibradas, una estrategia puramente aleatoria tendr?a una accuracy esperada cercana al 25 %. Por tanto, el Transformer preentrenado y ajustado mediante fine-tuning consigue m?s del doble de aciertos que el azar, lo que confirma que el modelo es capaz de capturar informaci?n relevante sobre la ideolog?a pol?tica a partir del contenido textual de los tweets.

# 7. Proyecto de extensi?n

An?lisis tem?tico de la desinformaci?n: Aplicar m?todos de modelado de t?picos o de clustering sobre embeddings contextuales para detectar temas recurrentes de desinformaci?n.

Para identificar los temas principales presentes en el corpus y detectar aquellos susceptibles de contener desinformaci?n o discursos polarizados, se ha aplicado el algoritmo K?Means sobre los embeddings obtenidos con Word2Vec. Este m?todo permite agrupar palabras que comparten contexto sem?ntico similar, revelando as? los bloques tem?ticos que estructuran el discurso pol?tico de los tweets.

Se ha evaluado si el algoritmo K?Means es capaz de encontrar similitudes entre palabras a partir de sus vectores de alta dimensi?n (200 dimensiones, definido en dimensi?n vector Word2Vec). Tambi?n, se ha decidido trabajar con 20 clusters, ya que un n?mero significativamente mayor incrementaba de forma notable el tiempo de ejecuci?n sin aportar mejoras sustanciales para el an?lisis. Con este valor se ha obtenido un equilibrio adecuado entre detalle tem?tico y eficiencia computacional.

A continuaci?n se muestra el resultado obtenido ordenado por clusters:

```
Cluster 0:
['resignaci?n', 'sr', 'asesinadas', 'escribo', 'xxi', 'decirlo', 'verlo', 'servir', 'adquisitivo', 'seguirlo', 'o?do', 'arcadi', 'imaginar', 'intentado', 'enteron', 'jaja', 'entrado', 'escuelas', 'presentamos', 'manipulaci?n']

Cluster 1:
['sido', 'nuevo', 'historia', 'ayer', 'ciudad', 'casa', 'importante', 'ejemplo', 'final', 'hilo', 'mundial', 'tarde', 'espa?old', 'informaci?n', 'primer', 'padre', 'mejores', 'alcalde', 'art?culo', 'vecinos']

Cluster 2:
['parte', 'presidente', 'nueva', 'adem?s', 'forma', 'europa', 'futuro', 'favor', 'realidad', 'real', 'proyecto', 'sector', 'pol?tico', 'espa?ola', 'situaci?n', 'paso', 'pueblo', 'centro', 'plan', 'hacia']

Cluster 3:
['manifestaci?n', 'sol', 'asesinato', 'asamblea', 'hermano', 'covid', 'presente', 'v', 'l?nea', 'recuperar', 'apenas', 'privado', 'mayo', 'apoyar', 'universidad', 'perdido', 'terrible', 'hablado', 'dato', 'burgos']

Cluster 4:
['muejejeje', 'apunto', 'alcaldesas', 'landaluce', 'publiqu?', 'cesiones', 'indignante', 'destruyen', 'avala', 'leerla', 'resuelto', 'diputadas', 'impulsado', 'jejeje', 'disfrutad', '000m', 'indepes', 'atreven', 'desea', 'reunidos']

Cluster 5:
['espa?a', 'madrid', 'frente', 'guerra', 'mientras', 'p?blico', 'sanidad', 'p?blica', 'mayor', 'empleo', 'dinero', 'grandes', 'medidas', 'comunidad', 'trabajadores', 'familias', 'inflaci?n', 'econom?a', 'salud', 'ayuntamiento']

Cluster 6:
['a?os', 'a?o', 'd?as', '000', '1', '2', '3', 'millones', '5', '4', 'euro', 'meses', '20', '10', '30', '6', '8', '7', '15', '9']

Cluster 7:
['q', 'pol?ticas', 'igualdad', 'cuentas', 'p?blicas', 'avanzar', 'web', 'seguir?', 'rindo', 'permitan']

Cluster 8:
['partido', 'caso', 'mujer', 'sigue', 'debate', 'medios', 'elecciones', 'acuerdo', 'posible', 'haciendo', 'siendo', 'cara', 'dicen', 'tipo', 'poner', 'calle', 'seguro', 'unas', 'lugar', 'camino']

Cluster 9:
['siempre', 'aqu?', 'as?', 'mejor', 'bien', 'ver', 'gente', 'tan', 'hecho', 'tiempo', 'creo', 'mundo', 'parece', 'cosas', 'pues', 'nunca', 'verdad', 'vamos', 'cuenta', 's?']

Cluster 10:
['democracia', 'seguir', 'tuit', 'quiero', 'compromiso', 'justa', 'comentarios', 'leo', 'defendiendo', 'ciudadano', 'marea', 'trumpista', 'agradecerlos']

Cluster 11:
['ah', 'pienso', 'veremos', 'contado', 'dep', 'raro', 'pierde', 'regalo', 'trabajadora', 'jajaja', 'empezamos', 'explico', 'candidatos', 'secretario', 'valent?a', 'local', 'credibilidad', 'explicado', 'obviamente', 'doy']

Cluster 12:
['politician', 'gobierno', 'ley', 'periodista', 'digital', 'congreso', 'poder', 's?lo', 'derecha', 'podemos', 'espa?oles', 'sino', 'cambio', 'impuestos', 'crisis', 'reforma', 'mayor?a', 'cgpj', 'constituci?n', 'constitucional']

Cluster 13:
['d?a', 'vez', 'cada']

Cluster 14:
['preciosa', 'preocupante', 'jajajaja', 'admiro', 'guay', 'infinito', 'pierden', 'quintero', 'catar', 'preparados', 'ajena', 'salva', 'pantalla', 'asistido', 'aprobamos', 'claras', 'bel?n', 'enga?ar', 'abrazos', 'exacto']

Cluster 15:
['bajando', 'comenzamos', 'jajaj', 'gustar', 'escr?pulos', 'gastar', 'habitante', 'confiar', 'twitcheada', 'refieres', 'tarda', 'colmo', 'parla', 'citar', 'preocupados', 'elemento', 'jose', 'tuitero', 'decisivo', 'andalucia']

Cluster 16:
['user', 'hashtag', 'hoy', 'gracias', 'gran', 'trabajo', 'muchas', 'ma?ana', 'amigo', 'abrazo', 'entrevista', 'enhorabuena', 'apoyo', 'junto', 'libro', 'familia', 'buen', 'equipo', 'noche', 'feliz']

Cluster 17:
['toda', 'persona', 'vida', 'pol?tica', 'pa?s', 'mujeres', 'social', 'derechos', 'hombre', 'violencia', 'justicia', 'derecho', 'libertad', 'v?ctimas', 'sociedad', 'seguridad', 'lucha', 'g?nero', 'instituciones', 'defender']

Cluster 18:
['si', 'ser', 'solo', 'ahora', 'puede', 'va', 'hacer', 'menos', 'c?mo', 'dice', 'mismo', 'decir', 'da', 'nadie', 'claro', 'tener', 'mal', 'aunque', 'igual', 'van']

Cluster 19:
['hace', 'do', 'despu?s', 'tras', 'primera', 'pasado', 'semana', 'casi', 'fin', 'datos', 'tres', 'medio', '2022', '2023', 'horas', 'lleva', 'hora', 'minutos', 'domingo', 'luz']
```
Los resultados muestran que los clusters obtenidos corresponden a tem?ticas bien diferenciadas. Por ejemplo, el Cluster 17 agrupa t?rminos relacionados con la violencia de g?nero y los derechos sociales ("mujeres", "violencia", "v?ctimas", "g?nero", "justicia"), mientras que el Cluster 5 re?ne palabras vinculadas al ?mbito econ?mico y a los servicios p?blicos ("sanidad", "inflaci?n", "empleo", "econom?a"). A nivel pol?tico-institucional, el Cluster 12 concentra vocabulario como "gobierno", "ley", "derecha", "constituci?n", indicando un bloque tem?tico m?s ideol?gico. Asimismo, el Cluster 3 contiene palabras asociadas a manifestaciones, sucesos y temas sensibles como "asesinato", "manifestaci?n" o "covid".

La existencia de estos bloques sem?nticos confirma que el clustering sobre embeddings permite identificar t?picos sensibles, muchos de los cuales coinciden con ?reas donde la desinformaci?n tiende a aparecer con mayor frecuencia como econom?a, pandemia, violencia o pol?tica institucional.

Como en Word2Vec, se ha realizado una representaci?n T-SNE:

![](/images/E4z_Image_21.png)

La visualizaci?n mediante t-SNE confirma que los embeddings han agrupado palabras de manera significativa: los clusters no est?n mezclados ca?ticamente, sino que presentan estructuras sem?nticas claras, con zonas de alta cohesi?n interna y separaciones naturales entre temas. Esto refuerza que la segmentaci?n por clusters es v?lida y ?til para entender la organizaci?n del vocabulario del corpus.

**8. Conclusi?n**

Este trabajo ha permitido explorar diversas t?cnicas de tratamiento de datos, destacando la importancia creciente de este proceso en un mundo cada vez m?s impulsado por la orientaci?n del dato. Actualmente, el tratamiento de datos se encuentra en auge, especialmente en el ?mbito empresarial, donde las organizaciones reconocen su valor estrat?gico para la toma de decisiones y la optimizaci?n de procesos. La existencia de datos coherentes e ?ntegros es una condici?n necesaria para que las empresas se puedan beneficiar de la inteligencia artificial, entre otras cosas.

Este trabajo demuestra c?mo un enfoque adecuado en el preprocesamiento de datos puede mejorar significativamente el rendimiento de modelos avanzados como Word2Vec, TF-IDF o BERT.
