from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from sklearn.base import clone


class ModifiedRank(DCSTechnique):
    def __init__(self, n_neighbors=7, algorithm="auto"):
        super().__init__(n_neighbors, algorithm)
        self.knn_dict = {}
        self.current_val_set_X_dict = {}
        self.current_val_set_y_dict = {}

    def fit(self, X, y):
        for i in np.unique(y):
            self.knn_dict[i] = clone(self.knn)
            filtered_X = X[y == i]
            filtered_y = y[y == i]
            self.knn_dict[i].fit(filtered_X, filtered_y)
            self.current_val_set_X_dict[i] = filtered_X
            self.current_val_set_y_dict[i] = filtered_y
        self.current_val_set_X = X
        self.current_val_set_y = y

    def estimate_competence(self, ensemble, instances):
        dict_neighbors_X = {}
        dict_neighbors_y = {}
        for class_, knn in self.knn_dict.items():
            neighbors_index = knn.kneighbors(instances, return_distance=False)
            dict_neighbors_X[class_] = self.current_val_set_X_dict[class_][neighbors_index, :]
            dict_neighbors_y[class_] = self.current_val_set_y_dict[class_][neighbors_index]
        consecutive_pred = np.empty((instances.shape[0], len(ensemble)), dtype=np.int)
        for index_clf, clf in enumerate(ensemble):
            for i in range(instances.shape[0]):
                predicted_class = clf.predict(instances[i:i+1])[0]
                predictions_neighbors = clf.predict(dict_neighbors_X[predicted_class][i])
                wrong_pred = predictions_neighbors != dict_neighbors_X[predicted_class][i]
                correct_cons = np.split(predictions_neighbors, np.where(wrong_pred)[0])[0]
                consecutive_pred[i, index_clf] = correct_cons.shape[0]

        competent_members = np.zeros((instances.shape[0], len(ensemble)), dtype=np.int)
        best_indexes = np.argmax(consecutive_pred, axis=1)
        for i in range(competent_members.shape[0]):
            competent_members[i][best_indexes[i]] = 1
        return competent_members
