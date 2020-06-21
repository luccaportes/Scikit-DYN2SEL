from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np


class KNORAU(DCSTechnique):
    def estimate_competence(self, ensemble, instances):
        neighbors_index = self.knn.kneighbors(instances, return_distance=False)
        neighbors_X = self.current_val_set_X[neighbors_index, :]
        neighbors_y = self.current_val_set_y[neighbors_index]
        competent_members = np.zeros((instances.shape[0], len(ensemble)), dtype=np.int)
        for index_clf, clf in enumerate(ensemble):
            for i in range(instances.shape[0]):
                predictions_neighbors = clf.predict(neighbors_X[i])
                n_correct_pred = np.sum(predictions_neighbors == neighbors_y[i])
                competent_members[i, index_clf] = n_correct_pred

        return competent_members
