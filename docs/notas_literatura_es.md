# Notas de literatura en espanol

Estas notas conectan las fuentes principales con el experimento del proyecto:
diversidad textual, tokens efectivos y generalizacion en modelos de lenguaje
pequenos de nivel caracter. No son un resumen exhaustivo de los libros; son una
guia de uso para escribir el informe sin sobreafirmar.

## Chang et al. 2024: tokens efectivos y calidad de datos

Referencia: Ernie Chang et al., "Scaling Parameter-Constrained Language Models
with Quality Data", EMNLP Industry Track 2024.

Uso central en el proyecto:

- El paper sirve para justificar que contar tokens crudos no basta para medir
  el valor de un corpus de entrenamiento.
- La idea de "tokens efectivos" permite formular la pregunta del proyecto:
  si dos corpus tienen el mismo tamano bruto, puede uno ser mas util por su
  diversidad o calidad?
- La adaptacion de este proyecto es deliberadamente reducida: se aisla una
  senal medible, la diversidad textual empirica, y se observa su relacion con
  perdida de test y brecha de generalizacion.

Puntos tecnicos que conviene usar:

- Chang et al. extienden la intuicion de leyes de escalamiento incorporando un
  termino de datos efectivos.
- Su nocion de calidad incluye diversidad y syntheticity estimada con un modelo
  docente. Este proyecto no replica esa parte: no usa modelo docente y no mide
  syntheticity.
- La comparacion local mantiene fijo el tamano bruto del corpus y el presupuesto
  de entrenamiento. Eso convierte el experimento en una prueba controlada de una
  sola dimension de calidad.
- La metrica `gzip_compression_ratio` en este repo se define como
  `compressed_size / original_size`; por tanto, valores mayores significan texto
  menos compresible y mas diverso empiricamente. Esto debe explicarse porque no
  coincide con todas las convenciones de razon de compresion.

Lectura para el informe:

- No decir que se validan las leyes de escalamiento de Chang et al.
- Si decir que el experimento esta inspirado en su marco conceptual.
- La conclusion negativa del piloto es compatible con el paper: mas diversidad
  no equivale automaticamente a mas tokens efectivos si el modelo, el
  optimizador o el presupuesto no pueden convertir esa diversidad en estructura.

Frase util:

> Inspirado por la perspectiva de tokens efectivos de Chang et al., el proyecto
> aisla una senal computable de calidad, la diversidad textual empirica, y mide
> si esta mejora la generalizacion cuando el tamano bruto del corpus permanece
> fijo.

## Petersen y Zech: teoria matematica de deep learning

Referencia: Philipp Petersen y Jakob Zech, "Mathematical Theory of Deep
Learning", arXiv:2407.18384.

Uso central en el proyecto:

- Da el marco formal de aprendizaje como minimizacion de riesgo empirico.
- Sirve para presentar la perdida de entrenamiento como una funcion objetivo
  sobre parametros `theta`.
- Permite separar tres niveles: aproximacion, optimizacion y generalizacion.
- Aporta lenguaje matematico para redes neuronales, backpropagation, descenso
  estocastico, Adam y arquitecturas modernas como Transformers.

Como aplicarlo al modelo de lenguaje:

Sea un corpus tokenizado

```text
x_0, x_1, ..., x_{N-1}.
```

Con tamano de bloque `B`, el dataset de entrenamiento se compone de ventanas
solapadas:

```text
X_i = (x_i, ..., x_{i+B-1})
Y_i = (x_{i+1}, ..., x_{i+B})
```

El modelo autoregresivo parametrizado por `theta` asigna probabilidades
condicionales del siguiente token:

```text
p_theta(x_{t+1} | x_{t-B+1}, ..., x_t).
```

La perdida usada en entrenamiento es entropia cruzada empirica:

```text
R_train(theta) = (1 / n) sum_i -log p_theta(Y_i | X_i).
```

La evaluacion compara:

```text
test_loss
test_perplexity = exp(test_loss)
generalization_gap = test_loss - train_loss
```

Advertencia metodologica:

- Las ventanas de lenguaje se solapan, asi que no son muestras iid.
- Por eso, los intervalos o errores por batch deben presentarse como resumen
  descriptivo, no como inferencia estadistica formal.
- Las cotas clasicas de generalizacion son utiles como marco conceptual, pero
  no explican por si solas el comportamiento observado en modelos pequenos.

## Calin: Deep Learning Architectures

Referencia: Ovidiu Calin, "Deep Learning Architectures: A Mathematical
Approach", Springer Series in the Data Sciences, 2020.

PDF local:
`data/pdfs/matematicas/llms/deep-learning-architectures/deep-learning-architectures.pdf`

Uso central en el proyecto:

- Complementa a Petersen-Zech con una presentacion matematica mas arquitectural
  de redes neuronales.
- Es especialmente util para activaciones, funciones de costo, optimizacion,
  backpropagation, redes recurrentes, informacion, entropia y capacidad.
- Conviene citarlo cuando se explique por que la entropia cruzada, el descenso
  de gradiente y las arquitecturas de red se pueden estudiar como objetos
  matematicos, no solo como implementaciones.

Elementos concretos recuperados del libro:

- El libro organiza la exposicion desde redes neuronales simples hasta
  aproximacion universal, procesamiento de informacion, geometria de redes y
  arquitecturas como CNN, RNN, GAN, Boltzmann machines y Hopfield networks.
- En funciones de activacion, distingue familias sigmoidales, funciones tipo
  "hockey-stick" como ReLU, ELU y SELU, y funciones tipo bump. Para este
  proyecto, ReLU/SwiGLU/RMSNorm no se deben presentar como detalles aislados:
  forman parte de una eleccion de parametrizacion que cambia la geometria de la
  funcion aprendida.
- En funciones de costo, el libro trata error cuadratico, entropia cruzada,
  divergencia KL, Jensen-Shannon y MMD. La entropia cruzada es la conexion mas
  directa con modelos de lenguaje, porque entrenar next-token prediction
  equivale a asignar alta probabilidad a los tokens observados.
- En optimizacion, presenta descenso de gradiente como movimiento en direccion
  de maximo descenso local, con tasa de aprendizaje `eta`; tambien cubre
  momentum, AdaGrad, RMSProp, Adam y AdaMax. Esto ayuda a justificar que AdamW
  no es un accesorio experimental sino una variante practica de una familia de
  metodos de minimizacion.
- La discusion de entrenamiento y error de test encaja con la comparacion del
  proyecto: un corpus puede producir menor perdida de entrenamiento pero peor
  perdida de test, o una mayor brecha, si el modelo memoriza regularidades que
  no transfieren bien.
- La parte de informacion y entropia es util para interpretar diversidad: mas
  entropia en el corpus significa mas incertidumbre empirica, pero no garantiza
  que esa incertidumbre sea aprendible con una red pequena.
- La seccion sobre RNNs conecta secuencias, perdida de informacion y gradientes
  desvanecientes. Aunque el proyecto usa Transformers, esta lectura sirve como
  contraste historico: las arquitecturas para secuencias surgen, en parte, para
  conservar y transformar informacion temporal.

Conexion con los resultados:

- El corpus de alta diversidad tiene mayor entropia y menor compresibilidad.
  Desde la perspectiva de Calin, eso puede verse como una entrada con mas
  informacion y mas patrones candidatos.
- Pero una red pequena tiene capacidad limitada y un presupuesto de optimizacion
  fijo. La diversidad adicional puede aumentar la dificultad predictiva antes
  de mejorar la generalizacion.
- Por eso, la conclusion debe formularse asi: diversidad empirica es una senal
  de riqueza del corpus, no una garantia de tokens efectivos.

## Wegner: Mathematical Introduction to Data Science

Referencia: Sven A. Wegner, "Mathematical Introduction to Data Science",
Springer, 2024.

PDF local:
`data/pdfs/matematicas/optimizacion/data science/data science.pdf`

Uso central en el proyecto:

- Provee una entrada rigurosa a data science desde regresion, k-NN, clustering,
  SVD, alta dimension, concentracion de medida, SVM, kernels, redes neuronales
  y descenso de gradiente.
- Es util para traducir el experimento a lenguaje de datasets, features,
  etiquetas, perdida y optimizacion.
- Ayuda a explicar por que alta dimension puede ser simultaneamente maldicion y
  bendicion: los datos se vuelven dispersos, pero tambien aparecen fenomenos de
  concentracion que hacen posibles separaciones y aproximaciones.

Elementos concretos recuperados del libro:

- Define datasets etiquetados y no etiquetados; en este proyecto, las ventanas
  `X_i` y objetivos `Y_i` forman un dataset supervisado de prediccion del
  siguiente caracter.
- Distingue tres perspectivas sobre aprender una funcion: aproximacion de una
  funcion verdadera, modelo probabilistico con ruido y minimizacion de una
  funcion de costo. El proyecto usa sobre todo la tercera perspectiva, aunque
  interpreta la entropia cruzada probabilisticamente.
- En regresion, muestra que minimizar cuadrados se conecta con maxima
  verosimilitud bajo ruido gaussiano. Esa idea general ayuda a explicar por que
  elegir una perdida implica tambien una hipotesis sobre el tipo de error o
  distribucion que se esta modelando.
- La SVD y las aproximaciones de bajo rango sirven como analogia para compresion
  y representaciones: no todo dato bruto aporta igual informacion; la pregunta
  es que estructura puede conservarse con menos parametros.
- Los capitulos de alta dimension y concentracion explican que, al aumentar la
  dimension, distancias y normas se comportan de forma poco intuitiva. Esto es
  relevante para lenguaje, donde los contextos y n-gramas crean espacios
  combinatorios grandes incluso a nivel caracter.
- El capitulo de perceptron muestra la transicion desde clasificadores lineales
  hacia redes. La limitacion del perceptron para funciones como XOR motiva
  arquitecturas con capas ocultas.
- El capitulo de redes neuronales incluye resultados de expresividad: redes
  suficientemente anchas o profundas pueden representar familias amplias de
  funciones. Para el proyecto, eso debe leerse con cuidado: expresividad
  teorica no implica que un modelo pequeno, entrenado pocos pasos, encuentre la
  funcion util.
- El capitulo de descenso de gradiente para funciones convexas da condiciones
  bajo las cuales la secuencia de valores decrece y converge. En redes profundas
  el problema no es convexo, pero el capitulo sirve como base para entender que
  el tamano de paso, la suavidad y la geometria local afectan el entrenamiento.

Conexion con los resultados:

- La comparacion medium vs high no es solo "mas datos variados es mejor"; es
  una comparacion entre dos distribuciones empiricas con diferente dificultad.
- La alta diversidad aumenta vocabulario, entropia y patrones locales. Eso puede
  elevar el error de test si el modelo no tiene capacidad suficiente o si el
  entrenamiento no alcanza a estabilizar representaciones.
- Wegner ayuda a escribir esta limitacion en terminos sobrios: el algoritmo
  aprende minimizando una perdida sobre un dataset concreto; no aprende una
  nocion abstracta de calidad de datos.

## Sintesis para el informe

Tesis prudente:

> Los resultados del piloto muestran que diversidad textual empirica y tokens
> efectivos no son sinonimos. La diversidad aumenta la riqueza estadistica del
> corpus, pero su valor depende de la capacidad del modelo, el presupuesto de
> optimizacion y la alineacion entre distribucion de entrenamiento y test.

Como usar cada fuente:

- Chang et al.: motivacion de tokens efectivos y datos de calidad.
- Petersen-Zech: marco de riesgo empirico, generalizacion y arquitecturas
  modernas.
- Calin: funciones de costo, activaciones, optimizacion e interpretacion
  informacional de redes.
- Wegner: formalizacion de datasets, perdida, alta dimension, redes y descenso
  de gradiente.

Limitacion que debe aparecer:

- El proyecto no prueba una teoria general de diversidad.
- El corpus es pequeno y caracter-level.
- Las ventanas se solapan.
- La evaluacion es empirica y descriptiva.
- Un resultado negativo contra "mas diversidad siempre ayuda" no contradice la
  idea de tokens efectivos; mas bien muestra que la efectividad es relativa al
  sistema completo de datos, modelo y entrenamiento.
