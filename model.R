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
task.goq = TaskRegr$new(id = "GOQ", backend = d.goq, target = "target")

d.gopi = subset(read.csv("GOPI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
task.gopi = TaskRegr$new(id = "GOPI", backend = d.gopi, target = "target")

d.pi = subset(read.csv("PI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
task.pi = TaskRegr$new(id = "PI", backend = d.pi, target = "target")

# run each learner on each task, with a 10-fold CV
design = benchmark_grid(
    tasks = list(task.goq, task.gopi, task.pi),
    learners = learners,
    resamplings = rsmp("cv", folds = 10)
)
bmr = benchmark(design)

# plot results
p = autoplot(bmr, measure = msr("regr.mae")) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1)) +
    expand_limits(y = 0) +
    geom_hline(data = data.frame(mean = c(0.394208, 0.550969, 0.370176), task_id = bmr$tasks$task_id),
               aes(yintercept = mean))
ggsave(p, file = "modeling-comparison.pdf", height = 5, width = 8)
