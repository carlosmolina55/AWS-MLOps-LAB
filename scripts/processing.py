import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. CARGAR EL DATASET

input_data_path = os.path.join("/opt/ml/processing/input", "dataset.csv") 
df = pd.read_csv(input_data_path)

# 2. FEATURE ENGINEERING

# A. Eliminar columnas que no aportan valor o son texto libre difícil de procesar
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')

# B. Imputar valores nulos (Missing values)
# Rellenamos las edades nulas con la mediana de las edades
df['Age'] = df['Age'].fillna(df['Age'].median())

# Rellenamos la tarifa nula (si la hubiera) con la mediana
if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Rellenamos el puerto de embarque nulo con el valor más frecuente (moda)
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# C. Codificar variables categóricas
# Convertimos 'Sex' a valores binarios (0 y 1)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding para 'Embarked' (Convierte el puerto en columnas numéricas separadas)
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# D. Crear nueva variable: Tamaño de la familia
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# E. Asegurar que todo sea numérico para evitar errores en algoritmos como XGBoost
df = df.astype(float)


# 3. DIVIDIR EL DATASET EN TRAIN, VALIDATION Y TEST

print("Shape of data is:", df.shape)
train, test = train_test_split(df, test_size=0.2)
train, validation = train_test_split(train, test_size=0.2)

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

try:
    print("Writing train file")
    train.to_csv("/opt/ml/processing/output/train/train.csv")
    print("Writing validation file")
    validation.to_csv("/opt/ml/processing/output/validation/validation.csv")
    print("Writing test file")
    test.to_csv("/opt/ml/processing/output/test/test.csv")
    print("Wrote files successfully")
except Exception as e:
    print("Failed to write the files")
    print(e)
    pass

print("Completed running the processing job")