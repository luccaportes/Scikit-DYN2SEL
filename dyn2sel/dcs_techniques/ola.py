from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np


class OLA(DCSTechnique):
    def estimate_competence(self, ensemble, instances):
        neighbors_index = self.knn.kneighbors(instances, return_distance=False)
        neighbors_X = self.current_val_set_X[neighbors_index, :]
        neighbors_y = self.current_val_set_y[neighbors_index]
        accuracies = np.empty((instances.shape[0], len(ensemble)), dtype=np.float)
        for index_clf, clf in enumerate(ensemble):
            for i in range(instances.shape[0]):
                predictions_neighbors = clf.predict(neighbors_X[i])
                accuracy = np.sum(predictions_neighbors == neighbors_y[i]) / \
                           predictions_neighbors.shape[0]
                accuracies[i, index_clf] = accuracy
        competent_members = np.zeros((instances.shape[0], len(ensemble)), dtype=np.int)
        best_indexes = np.argmax(accuracies, axis=1)
        for i in range(competent_members.shape[0]):
            competent_members[i][best_indexes[i]] = 1
        return competent_members