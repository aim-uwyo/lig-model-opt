#!/usr/bin/env Rscript

library(ggplot2)
library(scales)

library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3viz)
library(mlr3pipelines)

set.seed(123)

learners = list(lrn("regr.featureless"),
                lrn("regr.lm"),
                lrn("regr.ranger"),
                lrn("regr.km", nugget.stability = 1e-8, covtype = "powexp"),
                lrn("regr.ksvm"),
                lrn("regr.rpart"))
# wrap each learner in a one-hot encoder for categorical features (gas)
learners = lapply(learners, function(l) po("encode") %>>% po("learner", l))

# read data and remove meta-data columns
d.goq = subset(read.csv("GOQ.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
d.gopi = subset(read.csv("GOPI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
d.pi = subset(read.csv("PI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))

# train on one task, test on another, for every combination of tasks
task.goq.gopi = TaskRegr$new(id = "GOQ-GOPI", backend = rbind(d.goq, d.gopi), target = "target")
task.goq.pi = TaskRegr$new(id = "GOQ-PI", backend = rbind(d.goq, d.pi), target = "target")
task.gopi.goq = TaskRegr$new(id = "GOPI-GOQ", backend = rbind(d.gopi, d.goq), target = "target")
task.gopi.pi = TaskRegr$new(id = "GOPI-PI", backend = rbind(d.gopi, d.pi), target = "target")
task.pi.goq = TaskRegr$new(id = "PI-GOQ", backend = rbind(d.pi, d.goq), target = "target")
task.pi.gopi = TaskRegr$new(id = "PI-GOPI", backend = rbind(d.pi, d.gopi), target = "target")

indices.train = 1:(1*nrow(d.gopi))
indices.test = (1*nrow(d.gopi)+1):(2*nrow(d.gopi))

resampling = rsmp("custom")
resampling$instantiate(task.goq.gopi,
  # randomly choose 80% of training/test data, 10 times, for 10 train/test sets
  train = lapply(1:10, function(i) sample(indices.train, length(indices.train * 0.8))),
  test = lapply(1:10, function(i) sample(indices.test, length(indices.test * 0.8)))
)

# run each learner on each task, with the custom resampling that trains on one and evaluates on another task
design = benchmark_grid(
    tasks = list(task.goq.gopi, task.goq.pi, task.gopi.goq, task.gopi.pi, task.pi.goq, task.pi.gopi),
    learners = learners,
    resamplings = resampling
)
bmr = benchmark(design)

# plot results
p = autoplot(bmr, measure = msr("regr.mae")) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1)) +
    expand_limits(y = 0)
ggsave(p, file = "transfer-1-1.pdf", height = 8, width = 8)


# train on two tasks, test on the remaining
task.gen.goq = TaskRegr$new(id = "GOQ", backend = rbind(d.gopi, d.pi, d.goq), target = "target")
task.gen.gopi = TaskRegr$new(id = "GOPI", backend = rbind(d.goq, d.pi, d.gopi), target = "target")
task.gen.pi = TaskRegr$new(id = "PI", backend = rbind(d.gopi, d.goq, d.pi), target = "target")

indices.train = 1:(2*nrow(d.gopi))
indices.test = (2*nrow(d.gopi)+1):(3*nrow(d.gopi))

resampling = rsmp("custom")
resampling$instantiate(task.gen.goq,
  train = lapply(1:10, function(i) sample(indices.train, length(indices.train * 0.8))),
  test = lapply(1:10, function(i) sample(indices.test, length(indices.test * 0.8)))
)
design = benchmark_grid(
    tasks = list(task.gen.goq, task.gen.gopi, task.gen.pi),
    learners = learners,
    resamplings = resampling
)
bmr = benchmark(design)
p = autoplot(bmr, measure = msr("regr.mae")) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1)) +
    expand_limits(y = 0)
ggsave(p, file = "transfer-2-1.pdf", height = 5, width = 8)
