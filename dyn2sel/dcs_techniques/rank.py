from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np


class Rank(DCSTechnique):
    def estimate_competence(self, ensemble, instances):
        neighbors_distance, neighbors_index = self.knn.kneighbors(instances, return_distance=True)
        neighbors_X = self.current_val_set_X[neighbors_index, :]  # neighbors are returned sorted
        neighbors_y = self.current_val_set_y[neighbors_index]
        consecutive_pred = np.empty((instances.shape[0], len(ensemble)), dtype=np.int)
        for index_clf, clf in enumerate(ensemble):
            for i in range(instances.shape[0]):
                predictions_neighbors = clf.predict(neighbors_X[i])
                wrong_pred = predictions_neighbors != neighbors_y[i]
                correct_cons = np.split(predictions_neighbors, np.where(wrong_pred)[0])[0]
                consecutive_pred[i, index_clf] = correct_cons.shape[0]

        competent_members = np.zeros((instances.shape[0], len(ensemble)), dtype=np.int)
        best_indexes = np.argmax(consecutive_pred, axis=1)
        for i in range(competent_members.shape[0]):
            competent_members[i][best_indexes[i]] = 1
        return competent_members
