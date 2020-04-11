from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np


class KNORAE(DCSTechnique):
    def estimate_competence(self, ensemble, instances):
        selected_ensembles = np.ones((len(instances), len(ensemble))).astype(np.bool)
        for instance_index, instance in enumerate(instances):
            _, neighbors = self.knn.kneighbors(instance.reshape(1, -1))
            neighbors = neighbors.reshape(-1,)
            neighbors_X = self.current_val_set_X[neighbors, :]
            neighbors_y = self.current_val_set_y[neighbors]

            predictions = np.empty((len(ensemble), self.n_neighbors))
            for index, clf in enumerate(ensemble):
                predictions[index] = clf.predict(neighbors_X)
            n_corrects = -np.sum(predictions == neighbors_y, axis=1)
            _, counts = np.unique(np.sort(n_corrects), return_counts=True)
            selected_indexes = np.argsort(n_corrects)[:counts[0]]
            selected_ensembles[instance_index, selected_indexes] = False

        return selected_ensembles



