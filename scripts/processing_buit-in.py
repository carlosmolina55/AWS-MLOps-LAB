import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. CARGAR EL DATASET

input_data_path = os.path.join("/opt/ml/processing/input", "dataset.csv") 
df = pd.read_csv(input_data_path)

# 2. FEATURE ENGINEERING

# A. Eliminar columnas que no aportan valor o son texto libre difícil de procesar
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')

# B. Codificar variables categóricas
# Convertimos 'Sex' a valores binarios (0 y 1)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# C. Crear nueva variable: Tamaño de la familia
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print("Shape of processed data is:", df.shape)


# 3. DIVIDIR EL DATASET EN TRAIN, VALIDATION Y TEST

print("Shape of data is:", df.shape)
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, validation = train_test_split(train, test_size=0.2, random_state=42)

# Usamos .copy() para evitar SettingWithCopyWarning al modificar los subsets
train = train.copy()
validation = validation.copy()
test = test.copy()

# D. Imputar valores nulos (Missing values)
# Calculamos los estadísticos SOLO sobre train para evitar data leakage
# (no debe filtrarse información de validation/test a la imputación)
age_median = train['Age'].median()
embarked_mode = train['Embarked'].mode()[0] if 'Embarked' in train.columns else None

# Rellenamos las edades nulas con la mediana de las edades (de train)
for subset in (train, validation, test):
    subset['Age'] = subset['Age'].fillna(age_median)

# Rellenamos la tarifa nula (si la hubiera) con la mediana (de train)
if 'Fare' in train.columns:
    fare_median = train['Fare'].median()
    for subset in (train, validation, test):
        subset['Fare'] = subset['Fare'].fillna(fare_median)

# Rellenamos el puerto de embarque nulo con el valor más frecuente (moda de train)
if 'Embarked' in train.columns:
    for subset in (train, validation, test):
        subset['Embarked'] = subset['Embarked'].fillna(embarked_mode)

# E. One-Hot Encoding para 'Embarked' (Convierte el puerto en columnas numéricas separadas)
# Lo hacemos por subset y reindexamos validation/test a las columnas de train
if 'Embarked' in train.columns:
    train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
    validation = pd.get_dummies(validation, columns=['Embarked'], drop_first=True)
    test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)
    validation = validation.reindex(columns=train.columns, fill_value=0)
    test = test.reindex(columns=train.columns, fill_value=0)

# F. Asegurar que todo sea numérico para evitar errores en algoritmos como XGBoost
train = train.astype(float)
validation = validation.astype(float)
test = test.astype(float)

# 4. REESTRUCTURACIÓN PARA AWS BUILT-IN XGBOOST
# Aseguramos que 'Survived' es la primera columna (en cada subset)
if 'Survived' in train.columns:
    cols = ['Survived'] + [col for col in train.columns if col != 'Survived']
    train = train[cols]
    validation = validation[cols]
    test = test[cols]
else:
    print("Error: La columna 'Survived' no se encuentra en el dataset.")

# 5. CREACIÓN DE DIRECCTORIOS

try:
    os.makedirs("/opt/ml/processing/output/train")
    os.makedirs("/opt/ml/processing/output/validation")
    os.makedirs("/opt/ml/processing/output/test")
    print("Successfully created directories")
except Exception as e:
    # if the Processing call already creates these directories (or directory otherwise cannot be created)
    print(e)
    print("Could not make directories, already exist")
    pass

# 6. GUARDADO DE ARCHIVOS SIN HEADER Y SIN INDEX

try:
    print("Writing train file")
    # header=False es el cambio clave aquí para el built-in
    train.to_csv("/opt/ml/processing/output/train/train.csv", header=False, index=False)
    
    print("Writing validation file")
    validation.to_csv("/opt/ml/processing/output/validation/validation.csv", header=False, index=False)
    
    print("Writing test file")
    test.to_csv("/opt/ml/processing/output/test/test.csv", header=False, index=False)
    
    print("Wrote files successfully")
except Exception as e:
    print("Failed to write the files")
    print(e)
    pass

print("Completed running the processing job")