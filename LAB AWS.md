---
tags:
  - MLOps
  - ia
  - loyola
Fecha: 2026-06-17
profesor: Álvaro Martínez Ceballos
---
## LAB AWS

### Enunciado

**Preparación de datos y entrenamiento de modelo en Sagemaker**

En este LAB se hará una aplicación prática sobre dataset del Titanic con operaciones de limpieza y feature engineering pre-definidas, que pueden encontrarse en Moodle.

Se debe implementar un processing job que genere, al menos, los conjuntos de datos de entrenamiento y validación, a partir del conjunto de datos crudos, así como un entrenar un modelo, escogiendo libremente este una aproximación built-in o personalizada.

**Entregables**:

-          Enlace a un repositorio accesible con el código generado para ambos procesos

### Fase 1: Primeros pasos

1. Creación de repositorio en Github y clonación
2. Traducir el Moodle a un script de procesamiento (`preprocessing.py`)

### Fase 2: Entendiendo el código y la tarea

#### 1. Entender processing.py:

El script dado por el profesor `processing.py` lee un CSV de una ruta interna del contenedor, hace partición train, test y valid, y guarda los resultasoen las carpets de salida. `os.path.join` lo que hace es armar la ruta fina. La carpeta ya deberíavenirmontad. El CSV aparece porque el entorno de ejecución lo coloca en esa ruta. `os.makedirs` crear una ruta. Sagemaker montará los volúmenes de S3 en el contenedor de Scikit-learn.

#### 2.  Añado un poco de feature engineering al .py

Aunque no se si es necesario, voy a añadir un mínimo de operaciones de limpieza y feature engineering antes de hacer el train test split. He añadido: eliminar columnas que no aportan valor o son texto libre dificil de procesar, imputar valores nulos (edades con la mediana de las edades y puerto de embarque con el mas frecuente), codificamos variables categóricas (a binario para sex y one-hot-encoding para embarked), se crea nueva variable para tamaño de familia y por último se asegura que todo sea numérico para evitar errores en algoritmos de ML como XGBoost. 

#### 3. Ahora hay que ver como se integra en el notebook

En el notebook, se le está diciendo a SageMaker Processing que tome el archivo `processing.py` como el scipt que debe ejecutar dentro del job. El flujo es el siguiente:
1. El notebook llama a `sklearn_processor.run(...)`.
2. El parámetro `code="processing.py"` indica qué script local quiere enviar SageMaker.
3. SageMaker sube ese `.py` al contenedor del Processing Job.
4. Dentro del contenedor, ese script se ejecuta.
5. El script lee el input desde `/opt/ml/processing/input`.
6. El script escribe las salidas en `/opt/ml/processing/output/...`.
7. Luego SageMaker copia esas salidas al S3 que definiste en `ProcessingOutput`.

El noteboook no ejecuta directamente `processing.py` sino que lo envía al job y SageMaker lo ejecuta en un contenedor.

### Fase 3: Processing job en AWS

Ahora mismo estoy dentro de la interfaz de la consola de AWS. Tengo un perfil de usuario creado de cuando hicimos el ejercicio en clase y ese será el que use para la práctica para no crear uno nuevo. Usaré el dominio de SageMaker creado anteriormente:
![[Pasted image 20260618094408.png]]

Según entiendo no tengo que crear ahora mismo ningún bucket en S3 porque se supone que el propio sagemaker lo cogeá y slo subirá automáticamente a un bucket de S3 pordefecto y lueo mandarle esa ruta de S3 al Processing Job. 

Ahora creo una instancia para poder acceder a jupiter lab
![[Pasted image 20260618095235.png]]


Una vez hecho eso, desde jupyter me deja entrar en la interfaz. Como si fuera un VS Code
![[Pasted image 20260618095641.png]]

Ahora cargo el notebook, el .py y el dataset

![[Pasted image 20260618100134.png]]

Por lo visto este notebook estaba escrito para una versón antigua de sagemaker. Daba errores. Para igualar la versión a la del notebok se ejecuta al inicio del notebook lo siguiente:
`%pip install "sagemaker<3" --quiet`

Ya está ejecutando. Que está haciendo ahora mismo? esto:

1. Levanta una instancia `ml.m5.xlarge` (la que indicaste).
2. Descarga la imagen de contenedor de scikit-learn 1.2-1.
3. Copia tu `processing.py` y el input (`dataset.csv`) dentro del contenedor.
4. Ejecuta el script.
5. Sube los outputs (`train`, `validation`, `test`) a las rutas S3 que definiste.
6. Apaga la instancia.

Se puede ver como en S3  ha ido creando los buckets. En el de la foto esta `processing.py` y por otro lado `dataset.csv`

![[Pasted image 20260618102733.png]]

Y luego por otro lado podemos ver como guarda también el resultado de hacer el procesado correcctamente en tres sitios distintos:

![[Pasted image 20260618102921.png]]
Por ejemplo entrando a train:

![[Pasted image 20260618102953.png]]

Se ha completado por tanto con éxito el processing job.
![[Pasted image 20260618103202.png]]


### Fase 4: Realización del Training Job. Estructura del Dataset

Según el enunciado del LAB hay 2 formas de hacerlo: 
1. Algoritmo Built-in: Con contenedores pre-programados de SageMaker con algoritmos clásicos.
2. Script personalizado. Sería replicar lo que he hecho para el procesamiento.

Para aprender también como funcionaría crear un job con algoritmo ya preprogramado en sagemaker tiraremos por esta vía. 

#### Estructura del dataset

Para poder usar los algoritmos (XGBoost por ejemplo) de Sagemaker, un requisito importante es que se exige que el dataset no tenga cabecera (sin finla de nombres de columnas), que no tenga índices y que la variable objetivo esté estrictamente en la primera columna. 

Se crea para ello la nueva versión `processing_built-in.py` que aplica estas transformaciones al dataset. Reejecutamos el Job de processing en sagemaker y obtenemos los nuevos .csv
![[Pasted image 20260618115019.png]]

### Fase 5: Creación Notebook para Train Job

Existe un notebook para train dado por el profesor en moodle con built-in XGBoost. El problema es que este notebook de entrenamiento no esstá conectado automáticamente con la salida del processing. Ahora mismo depende de que se copien o se suban esos CSV. Es necesario modificar el notebook para que lea directamente lo que ha guardado el Job de Processing. 
Se modifica el código para que el job de entrenamiento se nutra de los csv generados y almacenados en S3 en el Job anterior de Processing

#### Mejoras del Job de entrenamiento en el notebook existente

Hacemos dos pequeñas mejoras en el noteboook:
1. Hacemos que XGBoost se base en la métrica `auc` para decidir si el modelo está mejorando. La métrica `auc` (área bajo la curva) es buena para clasificación binaria.
2. Añadimos early stopping. Si durante 10 rondas seguidas la métrica de validación no mejora se para el entrenamiento antes de llegar al número  máximo de épocas. Evita sobreentrenamiento y ahorra tiempo y coste de cómputo porque no se sigue entrenando innecesariamente. 
Se carga el notebook en jupyterlab y se ejecuta.

### Fase 6: Ejecución del Training Job 

Esta vez no ha dado problemas al princpio. Quizás pueda ser porque en el kernel ya cuenta como ejecutado `%pip install "sagemaker<3" --quiet` y no hace falta ejecutarlo otra vez. Cuidado porque si se reinicia el kernel y se ejecuta este sin haber ejecutado el notebook de processing antes puede que de error. 
Ejecutamos la celda para el training job: 

INFO:sagemaker:Creating training-job with name: sagemaker-xgboost-2026-06-18-10-23-37-759

```
2026-06-18 10:23:39 Starting - Starting the training job...
2026-06-18 10:23:54 Starting - Preparing the instances for training...
2026-06-18 10:24:42 Downloading - Downloading the training image......
2026-06-18 10:25:33 Training - Training image download completed. Training in progress.

...
[36]#011train-auc:0.91429#011validation-auc:0.80998
[37]#011train-auc:0.91429#011validation-auc:0.80998
[38]#011train-auc:0.91429#011validation-auc:0.80998
[39]#011train-auc:0.91498#011validation-auc:0.80470

2026-06-18 10:26:02 Uploading - Uploading generated training model
2026-06-18 10:26:02 Completed - Training job completed
Training seconds: 105
Billable seconds: 105
```

Ya tenemos el modelo entrenado. SageMaker indica que ha sido facturable por **105 segundos**

` [39]    train-auc:0.91498    validation-auc:0.80470`  indica que configuré auc como métrica. 

En el bucket de S3 podemos ver también el modelo almacenado
![[Pasted image 20260618123107.png]]


### Fase 7: Deploy y test

Creamos un endpoint para hacer deploy del modelo y evaluarlo. Levanta esa instancia y deja un servicio HTTP corriendo, escuchando peticiones de inferencia.
![[Pasted image 20260618123637.png]]

obtenemos las predicciones correctamente:
![[Pasted image 20260618123750.png|697]]

Y obtenemos la matriz de confusión

![[Pasted image 20260618123847.png|632]]

Por lo que se puede decir que se han cumplido los objetivos del lab satisfactoriamente. 

Por último importante cerrar el endpoint ya que si no seguiría consumiendo recursos y se quedaría facturando. 
![[Pasted image 20260618124047.png]]