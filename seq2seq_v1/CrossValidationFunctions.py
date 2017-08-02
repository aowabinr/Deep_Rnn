from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

space = {
         'units1': hp.uniform('units1', 64, 1024),
         'units2': hp.uniform('units2', 64, 1024),

         'dropout1': hp.uniform('dropout1', .25, .75),
         'dropout2': hp.uniform('dropout2', .25, .75),

         'batch_size': hp.uniform('batch_size', 28, 128),

         'nb_epochs': 100,
         'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
         'activation': 'relu'
         }