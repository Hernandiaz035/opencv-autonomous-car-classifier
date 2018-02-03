from __future__ import print_function, division
import numpy as np
import cv2
import xlrd
from time import time

#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
#from sklearn.metrics import confusion_matrix

#from sklearn.model_selection import cross_val_score

book_excel = xlrd.open_workbook("features_Imgs_Proyecto_Final.xlsx")



def load_xlsx(xlsx):
    
    sh = xlsx.sheet_by_index(0)
    x = np.zeros((sh.nrows,sh.ncols-1))
    y = []
    for i in range(0, sh.nrows):
        for j in range(0, sh.ncols-1):
            x[i,j] = sh.cell_value(rowx=i, colx=j+1)

        y.append(sh.cell_value(rowx=i, colx=0))
        
    y= np.array(y,np.float32)
    return x,y

##### Inicio del programa ######
if __name__ == '__main__':
    
    t0 = time()
    
    # Cargar datos desde un archivo .xlsx
    # la función retornará el número de muestras obtenidas y su respectiva clase
    X, Y = load_xlsx(book_excel)

    
##    robust_scaler = RobustScaler()
##    X = robust_scaler.fit_transform(X)


    standard_scaler = StandardScaler()    
    X = standard_scaler.fit_transform(X)

    joblib.dump(standard_scaler, 'Modelos_Predict/mlpScaler.pkl')
    
    # Se separan los datos: un % para el entrenamiento del modelo y otro
    # para el test
    total = []
    for i in range(10):
        samples_train, samples_test, responses_train, responses_test = \
                train_test_split(X, Y, test_size = 0.3)

   
        mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100,100), max_iter=1000, tol=0.0001)
    
        mlp.fit(samples_train, responses_train)    
        response_pred = mlp.predict(samples_test)
        total.append(accuracy_score(responses_test, response_pred)*100.0)
    
    print ("accuracy_score: ", mlp.score(samples_test, responses_test)*100.0)
    print ("Accuracy 10 iteraciones: ", total)
    print ("Promedio accuracy ", np.mean(total))
    print ("Matriz de confusion: \n",confusion_matrix(responses_test, response_pred))
    print ("\n")
    print("done in %0.16fs" % (time() - t0))

    joblib.dump(mlp, 'Modelos_Predict/mlp.pkl')
