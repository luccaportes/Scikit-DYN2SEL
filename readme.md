# Scikit-DYN2SEL
[![codecov](https://codecov.io/gh/luccaportes/Scikit-DYN2SEL/branch/master/graph/badge.svg?token=0R0BP0EOAJ)](https://codecov.io/gh/luccaportes/Scikit-DYN2SEL)
[![CircleCI](https://circleci.com/gh/luccaportes/Scikit-DYN2SEL.svg?style=shield&circle-token=c5d43eebff1c2b2d3e5e55f565a1ffde48136d1a)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
Scikit-DYN2SEL is a  framework  for  applying  and  implementing  DCS  techniques  in  the  data  stream  mining context.

## Features

### Implemented on top of scikit-multiflow
Scikit-DYN2SEL fully works with ensembles from the scikit-multiflow library, or any implementation that follows the same interface.

### User-friendly
Scikit-DYN2SEL classifiers follows the same interface as scikit-multiflow, which is based on the popular scikit-learn. Thus, scikit-DYN2SEL will be extremely familiar if you know any of these libraries.

### Stream learning tools
In its current state, scikit-multiflow contains data generators, multi-output/multi-target stream
learning methods, change detection methods, evaluation methods, and more.

### Open source and open to contributions
Distributed under the [MIT license]("https://github.com/luccaportes/Scikit-DYN2SEL/blob/master/LICENSE"), scikit-DYN2SEL implements the current state of art methods, however if you thinks it that it misses a method, feel free to either open an issue or opening a pull request with your implementation.

## Usage
The usage of Scikit-DYN2SEL is very straightforward.
```python
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from dyn2sel.apply_dcs import DYNSEMethod
from dyn2sel.dcs_techniques import KNORAE

clf = DYNSEMethod(
    HoeffdingTree(), chunk_size=1000, 
    dcs_method=KNORAE(), max_ensemble_size=10)
gen = SEAGenerator()
ev = EvaluatePrequential()
ev.evaluate(gen, clf)
```
## Citation
If you use scikit-dyn2sel in a paper, please cite us as following:
```
@ARTICLE{dyn2sel,
       author = {{Portes Cavalheiro}, Lucca and {Barddal}, Jean Paul and
         {Souza de Britto Jr}, Alceu and {Heutte}, Laurent},
        title = "{scikit-dyn2sel -- A Dynamic Selection Framework for Data Streams}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning},
         year = 2020,
        month = aug,
          eid = {arXiv:2008.08920},
        pages = {arXiv:2008.08920},
archivePrefix = {arXiv},
       eprint = {2008.08920},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200808920P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
