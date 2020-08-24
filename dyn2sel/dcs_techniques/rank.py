from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np


class Rank(DCSTechnique):
    """
    DCS-RANK
    The DCS-RANK method ranks the classifiers by competence and selecting the first position of the rank to predict the
    instance. The method starts with gathering the neighbors using K-Nearest Neighbors, and these are sorted according
    to the Euclidean distance to the instance to be predicted. Next, the rank of competence of the classifiers is built,
    based on the number of consecutive correct predictions each classifier made on the list of neighbors.

    Parameters
    ----------

    References
    ----------
        Sabourin, M., Mitiche, A., Thomas, D., Nagy, G. 1993. Classifier combination forhand-printed digit recognition.
        In:Proceedings of 2nd International Conference on Document Analysis and Recognition (ICDAR ’93). [S.l.: s.n.].
        p. 163–166.
    """

    def _estimate_competence(self, ensemble, instances):
        neighbors_distance, neighbors_index = self.knn.kneighbors(
            instances, return_distance=True
        )
        neighbors_X = self.current_val_set_X[
            neighbors_index, :
        ]  # neighbors are returned sorted
        neighbors_y = self.current_val_set_y[neighbors_index]
        consecutive_pred = np.empty((instances.shape[0], len(ensemble)), dtype=np.int)
        for index_clf, clf in enumerate(ensemble):
            for i in range(instances.shape[0]):
                predictions_neighbors = clf.predict(neighbors_X[i])
                wrong_pred = predictions_neighbors != neighbors_y[i]
                correct_cons = np.split(predictions_neighbors, np.where(wrong_pred)[0])[
                    0
                ]
                consecutive_pred[i, index_clf] = correct_cons.shape[0]

        competent_members = np.zeros((instances.shape[0], len(ensemble)), dtype=np.int)
        best_indexes = np.argmax(consecutive_pred, axis=1)
        for i in range(competent_members.shape[0]):
            competent_members[i][best_indexes[i]] = 1
        return competent_members
