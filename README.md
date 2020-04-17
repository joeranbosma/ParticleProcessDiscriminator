# Particle Process Discriminator
This projects explores several classifiers to discriminate between signal particle processes (involving 4 top quarks in our case) and background processes (involving a top and antitop quark). The same analysis can also be applied to distinguish between different particle processes. 

1. This project start with some exploratory data analysis in:  
`Data Exporation and Preprocessing.ipynb`  

2. Several conventional machine learning classifiers are tested in:  
`Algorithm Selection.ipynb`  
Tested are: single Decision Trees, boosted ensembles of Decision Trees, bagged ensembles of Decision Trees and Random Forest classifiers.  

3. Performance of Deep Neural Networks is optimised in:  
`Train Deep Neural Networks.ipynb`  
Also, rotational symmetry and inversion symmetry of particle processes is incorporated here.  

4. Convolutional Neural Networks are employed to better extract spatial relations in:  
`Train Convolutional Neural Networks.ipynb`  
Additionally, ensembles of Convolutional Neural Networks are used to boost performance. 

## Report
The most important findings of this project are presented in a report: [Distinguishing_4_top_events_from_background.pdf](https://github.com/joeranbosma/ParticleProcessDiscriminator/glob/master/Distinguishing_4_top_events_from_background.pdf). 

## Data
The used dataset is a subset of a generated LHC-like dataset as part of [1], and is available at https://www.phenomldata.org/. The subset consists of 83.300 ttbar events and 16.700 4top events. The used subset is available in [TrainingValidationData.csv](https://github.com/joeranbosma/ParticleProcessDiscriminator/glob/master/TrainingValidationData.csv).

[1] G. Brooijmans, A. Buckley, S. Caron, et. al. Les Houches 2019 Physics at TeV Colliders: New Physics Working Group Report. [arXiv:2002.12220](https://arxiv.org/abs/2002.12220), Feb 2020.
