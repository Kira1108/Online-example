from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def get_model(model_name):
    models = {
    "KNN":KNeighborsClassifier,
    "SVM":SVC,
    "Random Forest":RandomForestClassifier
    }
    
    return models.get(model_name)