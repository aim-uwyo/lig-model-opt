#!/usr/bin/env python3

import autosklearn.regression
import sys
import pandas as p

d = p.read_csv(sys.argv[1])
df = p.DataFrame(d)
# gas is a categorical feature
df['gas'] = df['gas'].astype('category').cat.codes
# remove meta-data columns
df = df.drop(['campaign', 'initial'], axis = 1)

model = autosklearn.regression.AutoSklearnRegressor(
        resampling_strategy = "cv",
        resampling_strategy_arguments = {'folds': 10},
        metric = autosklearn.metrics.mean_absolute_error,
        seed = 1,
        # run for one hour
        time_left_for_this_task = 3600
    )
# the last column is the target, everything else features
model.fit(df[df.columns[:-1]], df[df.columns[-1]])

print(model.sprint_statistics())
print(model.show_models())
