#!/usr/bin/env Rscript

library(ggplot2)
# mlr3mbo isn't ready yet, so we use the old version for now
library(mlr)
library(mlrMBO)

# read data and remove meta-data columns
d.goq = subset(read.csv("GOQ.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
d.gopi = subset(read.csv("GOPI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))
d.pi = subset(read.csv("PI.csv", stringsAsFactors = TRUE), select = -c(campaign, initial))

mkMBO = function(data, title) {
    # train the model we will optimize on
    model = train(makeLearner("regr.ranger"), makeRegrTask(data = data, target = "target"))

    # build optimization function, based on model we trained above
    fun = function(x) {
        df = as.data.frame(x)
        # only one type of gas will be selected, need to make sure that all levels are present
        df$gas = factor(df$gas, levels = levels(data$gas))
        return(getPredictionResponse(predict(model, newdata = df)))
    }

    # parameter set to optimize over
    ps = makeParamSet(
      makeIntegerParam("power", lower = 10, upper = 5555),
      makeIntegerParam("time", lower = 500, upper = 20210),
      makeDiscreteParam("gas", values = c("Argon", "Nitrogen", "Air")),
      makeIntegerParam("pressure", lower = 0, upper = 1000)
    )

    # wrap objective function for mlrMBO
    objfun = makeSingleObjectiveFunction(
         name = title,
         fn = fun,
         par.set = ps,
         has.simple.signature = FALSE,
         minimize = FALSE
    )

    # seed affects the initial data -- if very good points are sampled no improvement during the optimization can be seen
    set.seed(6)
    # sample 9 points for the initial data, stratified across gases
    samples.argon = sample(rownames(data[data$gas == "Argon", ]), 3)
    samples.nitro = sample(rownames(data[data$gas == "Nitrogen", ]), 3)
    samples.air = sample(rownames(data[data$gas == "Air", ]), 3)
    initial.data = data[c(samples.argon, samples.nitro, samples.air), ]

    ctrl = makeMBOControl(y.name = "target")
    ctrl = setMBOControlInfill(ctrl)
    # run for 50 iterations to demonstrate process
    ctrl = setMBOControlTermination(ctrl, iters = 50)

    # run Bayesian optimization
    res = mbo(objfun, design = initial.data, control = ctrl, show.info = TRUE)
    # extract cumulative objective value over optimization iteration for plotting
    opt.path = data.frame(target = cummax(getOptPathY(res$opt.path))[11:60])
    p = ggplot(data = opt.path, aes(x = 1:nrow(opt.path), y = target)) +
        geom_line() +
        xlab("Iteration") +
        # show initial data as boxplot at iteration 0
        geom_boxplot(data = data.frame(target = getOptPathY(res$opt.path)[1:10]),
                     aes(x = 0, y = target),
                     outlier.size = 1)
    ggsave(p, file = paste0("MBO-", title, ".pdf"), height = 5, width = 8)
}

# run optimization and save plots for all datasets
mkMBO(d.gopi, "GOPI")
mkMBO(d.goq, "GOQ")
mkMBO(d.pi, "PI")
