from dyn2sel.apply_dcs import WPSMethod, WPSMethodPostDrift
from skmultiflow.meta import AdaptiveRandomForest, OzaBagging
from skmultiflow.data import SEAGenerator, AGRAWALGenerator, ConceptDriftStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from copy import deepcopy

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
import multiprocess

generators = [
    (ConceptDriftStream(
        SEAGenerator(random_state=42, classification_function=0),
        SEAGenerator(random_state=42, classification_function=1),
        position=50000, width=1, random_state=42
    ), "SEA_CONCEPT"),
    (ConceptDriftStream(
        AGRAWALGenerator(random_state=42, classification_function=0),
        AGRAWALGenerator(random_state=42, classification_function=1),
        position=50000, width=1, random_state=42
    ), "AGRAWAL_CONCEPT")
]

classifiers = [
    (AdaptiveRandomForest(n_estimators=10, random_state=42), "ARF_10"),
    (AdaptiveRandomForest(n_estimators=100, random_state=42), "ARF_100"),
    (OzaBagging(base_estimator=HoeffdingTree(), n_estimators=10, random_state=42), "OZA_10_HT"),
    (OzaBagging(base_estimator=HoeffdingTree(), n_estimators=100, random_state=42), "OZA_100_HT"),
    (OzaBagging(base_estimator=NaiveBayes(), n_estimators=10, random_state=42), "OZA_10_NB"),
    (OzaBagging(base_estimator=NaiveBayes(), n_estimators=100, random_state=42), "OZA_100_NB"),
]
wps_classifiers = []
sizes = [10, 100] * 3
count = 0
for clf, name in classifiers:
    wps_classifiers.append((WPSMethod(deepcopy(clf),
                                      n_selected=int(sizes[count] * 0.4),
                                      window_size=5000,
                                      metric="accuracy"),
                            "WPS_ACC_" + name)
                           )
    count += 1

count = 0

for clf, name in classifiers:
    wps_classifiers.append((WPSMethod(deepcopy(clf),
                                      n_selected=int(sizes[count] * 0.4),
                                      window_size=5000,
                                      metric="recall"),
                            "WPS_REC_" + name)
                           )
    count += 1

count = 0
for clf, name in classifiers:
    wps_classifiers.append((WPSMethodPostDrift(deepcopy(clf),
                                               n_selected=int(sizes[count] * 0.4),
                                               window_size=5000,
                                               metric="accuracy", accuracy_gain_size=500),
                            "WPS_POST_DRIFT_ACC_" + name)
                           )
    count += 1

count = 0

for clf, name in classifiers:
    wps_classifiers.append((WPSMethodPostDrift(deepcopy(clf),
                                               n_selected=int(sizes[count] * 0.4),
                                               window_size=5000,
                                               metric="recall", accuracy_gain_size=500),
                            "WPS_POST_DRIFT_REC_" + name)
                           )
    count += 1

classifiers = classifiers + wps_classifiers

# classifiers = wps_classifiers

args_list = []
for gen in generators:
    for clf in classifiers:
        args_list.append((deepcopy(clf), deepcopy(gen)))


def run(args, sema):
    try:
        sem.acquire()
        clf_args, gen_args = args
        clf, clf_name = clf_args
        gen, gen_name = gen_args
        print("running", clf_name, gen_name)
        gen.prepare_for_use()
        filename = "generator=" + gen_name + "&clf=" + clf_name + ".csv"

        ev = EvaluatePrequential(n_wait=10000, max_samples=100000,
                                 pretrain_size=0,
                                 metrics=["accuracy", "model_size", "running_time"],
                                 output_file="results_drift/" + filename)
        ev.evaluate(stream=gen, model=clf)
        sema.release()
        print("finished", clf_name, gen_name)
    except:
        sema.release()
        print("error with", clf_name, gen_name)


# args_list = args_list[:4]
# p = ProcessingPool(1)
# p.map(run, args_list, chunksize=1)
sem = mp.Semaphore(mp.cpu_count())
p_list = []
for i in args_list:
    p = mp.Process(target=run, args=(i, sem))
    p.start()
    p_list.append(p)

for i in p_list:
    i.join()

# multiprocess.Pool(mp.cpu_count()).map(run, args_list)
