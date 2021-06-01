# Modeling and Optimizing Laser-Induced Graphene

This repository contains the data and code to model and optimize the production
of laser-induced graphene. All data were obtained in the same experimental setup
and environment, but from different precursor materials:
- GOPI.csv from graphene oxide on polyimide,
- GOQ.csv from graphene oxide on quartz,
- PI.csv from polyimide directly.

All three files give the target value (the G/D ratio that quantifies the
reduction of the precursor material into graphene, larger is better) for the
chosen experimental parameters laser power in mW, time the laser was applied to
a spot in ms, the gas in the reaction chamber, and the pressure in the reaction
chamber in psi. One row corresponds to one experiment. In some cases, the
experiment resulted in the destruction of the material or no effect at all was
observed; we set the target ratio to 0 in those cases.

The data was obtained from runs of Bayesian optimization with the aim to
maximize the target G/D ratio -- after a small set of initial, random
experiments, the parameter configurations to evaluate were chosen by the
optimization process. In particular, this means that the data are not
independent and violate the i.i.d. assumption.

The final two columns designate, for each experiment, which experimental run it
belongs to ("campaign") and whether it was part of the initial, random parameter
configurations, or chosen by the Bayesian optimization. An experimental campaign
was run on a single sample; different campaigns were run on different samples
that were prepared using the same method with the same precursor material.

The following code files are provided to explore different aspects of the data:
- model.R builds and compares some basic machine learning models on the
  datasets, including a dummy model that predicts the mean value of the training
  data. It produces the following output:
    - modeling-comparison.pdf
- autosklearn.py runs auto-sklearn on the given dataset for an hour to give an
  impression of what can be achieved with automated machine learning approaches.
  The results are shown in the PDF produced by model.R; the raw output are
  available in the following files:
    - GOPI-autosklearn.out
    - GOQ-autosklearn.out
    - PI-autosklearn.out
- transfer.R builds and compares some basic transfer learning models, where
  machine learning approaches are trained on datasets one or more types of
  precursor materials and evaluated on another. It produces the following files:
    - transfer-1-1.pdf shows results from training on one dataset and evaluating
      on another, for all combinations of datasets.
    - transfer-2-1.pdf shows results of training on two datasets and evaluating
      on the remaining one.
- mbo.R builds a model of how a particular precursor material reacts to
  experimental conditions based on the entire data and then runs Bayesian
  optimization on this model. This file also encodes the parameter space for the
  four experimental parameters. It produces the following files, showing the
  improvement over optimization iteration for each of the precursor materials:
    - MBO-GOPI.pdf
    - MBO-GOQ.pdf
    - MBO-PI.pdf
