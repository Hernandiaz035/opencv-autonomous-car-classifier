import numpy as np
import cv2
import xlrd
from time import time

#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.externals import joblib

from sklearn import preprocessing
from sklearn import decomposition

from sklearn.preprocessing import StandardScaler, RobustScaler


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

    total = []
    for i in range(10):
        #Se escalizan los valores a una distribución normal.
        standard_scaler = StandardScaler()    
        X_scaled = standard_scaler.fit_transform(X)

        joblib.dump(standard_scaler, 'Modelos_Predict/svmScaler.pkl')
        
        # Se separan los datos: un % para el entrenamiento del modelo y otro
        # para el test
        samples_train, samples_test, responses_train, responses_test = \
                    train_test_split(X_scaled, Y, test_size = 0.3)

        #Modelo con los datos escalados
##        print "Modelo con los datos escalados"
        kernel = 'linear'#'linear', 'rbf', 'poly'
        svm =SVC(C=1, kernel=kernel, degree=2 , tol = 1e-6)
        svm.fit(samples_train,responses_train)

        response_pred = svm.predict(samples_test)
        
##        print "accuracy_score: ", svm.score(samples_test, responses_test)
        total.append(accuracy_score(responses_test, response_pred)*100.0)
        scores = cross_val_score(svm, samples_train,responses_train, cv=10, scoring='accuracy')
        #print scores
        #print scores.mean()
    print ("Accuracy 10 iteraciones: ", total)
    print ("Promedio accuracy ", np.mean(total))
    print "Matriz de confusion: \n",confusion_matrix(responses_test, response_pred)

    joblib.dump(svm, 'Modelos_Predict/svm.pkl')
