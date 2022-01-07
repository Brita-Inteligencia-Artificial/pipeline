# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/TRABAJO ARIHUS IA/PROGRAMAS CURSO IA/PARTE 1/P14-Part1-Data-Preprocessing/Section 3 - Data Preprocessing in Python/Python/Data.csv')
print("dataser".center(150, "-"))
print(dataset)
X = dataset.iloc[:, :-1].values # Extrae todos los valores del dataset hasta la penultima columna
y = dataset.iloc[:, -1].values  # Extrae solamente la ultima columna, la etiqueta, que es lo mismo que y = dataset.iloc[:, 3].values
print("X".center(150, "-"))
print(X)
print("y".center(150, "-"))
print(y)
########################################################################### Tratamiento de los datos NaN ###################################################################################
# Taking care of missing data
from sklearn.impute import SimpleImputer   # SimpleImputer es una libreria para ralizar preprocesado de los datos
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#, axis = 0)
# Se crea el objeto de la clase, el parmetro "missing_values=np.nan" sirve para saber cuales son los datos que deben ser detectados como desconocidos,
#     en este caso la palabra que tendremos en un dataset cuando tendremos un valor desconocido sera un "nan"
# El segundo parametro es "strategy='mean'" se trata de como remplazaremos el valor desconocido, en este caso es sustituirlo por la media de la columna,
# puede ser igual la mediana u otro valor
# Por ultimo "axis= " es para indicar si vamos a sustituir ese valor por la media, de la fila o de la columna, en este caso seleccionaremos la
# media de la columna, ademas cuando queremos seleccionar algo de una columna colocamos "axis= 0", sigunifica aplica la funcion media a los valores desconocidos
#    por columna, si queremos que haga esta funcion por fila entonces solo cambiamos en 0 por el 1


imputer = imputer.fit(X[:, 1:3])    # "imputer.fit" sirve para arreglar la matriz de datos X de la fila 1  a la 3, sobreescribimos en la variable "imputer", y en pocas palabras lo
                                    #    utilizamos para sobreescribir nuestro "imputer" porque le acabamos de indicar las directrices con las que debe trabajar, debe ajustar los
                                    #    valores desconocidos a partir de ese conjunto de datos y por ultimo lo que hace es sobreescribir los valores desconocidos de la matriz
                                    #    de datos "X"
X[:, 1:3] = imputer.transform(X[:, 1:3])   # de X, para todas las filas de la 1 hasta la 3, lo igualamos a "imputer.transform"  que este ultimo se encargara de con la misma matriz de
                                           #    datos X, para todas las filas de 1 a 3 devolvera y sustituira con la asignacion los valores desconocidos (NaN) que aparecen dentro de la
                                           #    matriz de caracteristica inicial, entonces aplicamos la transformacion de los valores desconocidos de "X" a traves de que el
                                           #    "imputer.transform" transforme el conjunto de datos original
print("X con valores NaN sustituidos".center(150, '-'))
print(X)
################################################################## "variable categorica" y la "variable ordinal"#####################################################################
############################################################################## " Tipo de dato Nummpy" #####################################################################
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# # Encoding the Dependent Variable o codificacion de la variable dependiente
# from sklearn.preprocessing import LabelEncoder
# labelencoder_X = LabelEncoder()     # Crea un codificador de datos
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # "le.fit_transform" toma directamnte las columnas que yo le indique, en este caso la columna 0, que son categorias y los transforma a
#                                                        #    valores numericos, en este caso en 0, 1 y 2, modificando la columna 0 de mi dataset
# print("labelencoder_X".center(150, '-'))               # El problema al hacer esto es que le da un valor numerico a cada pais, y al trabajar con esto en un modelo de machine learning
# print(X)                                               #    el sistema pensara que por  decir un ejemplo, 0 es menor que 2, diciendo que francia es menor que españa, caso que no es cierto
                                                         #    son variables no comparables, de modo que esta traduccion no es correcta
# la diferencia entre la "variable categorica" y la "variable ordinal", una variable ordinal es una categoria que tiene cierto orden, como la talla de la ropa, sin embargo, para
#   categorias puras y duras donde no existe un orden, variables que no son ordinales
# Esta traduccion que hemos hecho no es correcta
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Variable Dummy o variable "OneHotEncoder" es una forma de traducir una categoria que no tiene un orden, no es una variable Ordinal si no que es puramente categorica, a un conjunto
#    de tantas columnas como categorias existen, en este caso tenemos 3 etiquetas diferentes para la categoria Francia, España y Alemania, entonces el vector va a ser traducido a 3
#    columnas, una para cada una de las etiquetas de la categoria, una para francia, una para España y otra para Alemania

# Encoding categorical data o Codificar datos categoricos, realiza una "traduccion numerica" de categorias escritas, como en este caso, de paises
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder   # Sirve para crear variables Dummy o OneHotEncoder (codificacion de un solo uno por fila)
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.float)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # Como argumento le debemos de pasar un parametro que es en que columna de
                                                                                                  #      nuestro dataset se encuentra la variable categorica que queremos
                                                                                                  #      convertir en Dummy, evidentemente se trata de la columna numero 0
X = np.array(ct.fit_transform(X)) # Codificamos X despues de aplicarle "ct.fit_transform" para ralizar la transformacion de datos, y como lo queremos transformar a vector
                                                #      columna aplicamos "np.array"
print("X con valores NaN sustituidos y datos de paises categorizados".center(150, '-'))
print(X)

# Encoding the Dependent Variable o codificacion de la variable dependiente
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()     # Crea un codificador de datos
y = le.fit_transform(y) # "le.fit_transform" toma directamnte las columnas que yo le indique, que son categorias y los transforma a valores numericos, en este caso en 0 y el 1
print("y categorizado".center(150, '-')) # aqui si podriamos ocupar el proceso de "le.fit_transform" porque solo hay dos valores, entoncs el resultante podria ser un valor binario
print(y)

############################################################### "Division de datos de entrenamiento y datos de prueba" ################################################################

# Splitting the dataset into the Training set and Test set o dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # Toma los datos "X" y los datos "y", los divide en 20% datos de prueba o test, y
                                                                                              #    "random_state = 1" que se trata del numero para poder reproducir el algoritmo, como esta
                                                                                              #    division de datos sera aleatoria en principio cada ves que la ejecutemos, dara como resultado
                                                                                              #    un valor diferente, entonces para que no ocurra eso y para que siempre devuelva el mismo
                                                                                              #    resultado colocamos una semilla aleatoria, un numero que si colocamos el mismo (en este
                                                                                              #    ejemplo el numero 0) nos dara siempre el mismo resultado, puede ser cualquier numero
print("X_train o datos de entrenamiento".center(150, '-'))
print(X_train)
print("X_test o datos de prueba".center(150, '-'))
print(X_test)
print("y_train o datos de entrenamiento".center(150, '-'))
print(y_train)
print("y_test o datos de prueba".center(150, '-'))
print(y_test)

################################################################################### "Escalado de variables" #################################################################################

# Feature Scaling o escalada de valores para poder normalizarlos para que esten definidos en un mismo rango de valores, esa normalizacion o estandarizacion de valores es muy
#   importante porque esto evita que algunas variables dominen sobre otras dentro de nuestro algoritmo de Machine learning y que el propio algoritmo sea el que deba disernir entre
#   que peso dar a cada una de las variables, no por tener un rango pequeño o grande, si no porque puede aportar algo en el proceso de prediccion o de clasificacion de nuestros datos

# Existen dos conceptos muy importantes, la "ESTANDARIZACION" y la "NORMALIZACION"

# Xstand =     X - mean(X)                  Xnorm =      X - min(X)
#         standard deviation (X)                       max(X) - min(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# No convertiremos las variables DUMMY (que son las filas de la 0 a la 2)
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])   # Escalamos automaticamente el conjunto de entrenamiento "X_train" para [:, 3:] (de la fila 3 hasta adelante)
X_test[:, 3:] = sc.transform(X_test[:, 3:])         # el conjunto "X_test" (de la fila 3 hasta adelante) se va a escalar, pero en lugar de ser un ".fit_transform" solo invocamos el
                                                    #    ".transform" para que escale los datos de "X_test" con la misma transformacion que haya detectado o que tiene que hacer a partir
                                                    #    de los datos de entrenamiento
print("X_train o datos de entrenamiento del escalamiento automatico".center(150, '-'))
print(X_train)
print("X_test o datos de prueba del escalamiento automatico".center(150, '-'))
print(X_test)