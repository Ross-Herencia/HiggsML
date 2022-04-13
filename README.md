# HiggsML

HiggsML is part of a phyiscs undergraduate research project undertakan at Queen Mary University of London.

Machine learning techniques are applied to the problem of classifying data from the Large Hadron Collider (LHC) at CERN. The aim is to develop a model that is able to classify signals of a heavy Higgs Boson, *A*, from a background. A neural network classifier is implemented in PyTorch and trained on approximately two million simulated samples, over a range of values for the mass of *A*. The mass of the *A* boson which is used to generate the simulation data is included as an input feature to create a parameterised neural network, which has previously been shown to improve a models' ability to generalise.
