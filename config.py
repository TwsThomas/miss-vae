# defnie all arguments for the experiences.
# so we have all when analyzing results.

args = dict()

args = {
    'model' : "dlvm",
    'n' : 1000,
    'd' : 3,
    'p' : 100,
    'prop_miss' : 0.1,
    'seed'  : 0,
    'd_miwae' : 3,
    'n_epochs' : 602,
    'sig_prior' : 1,
    'method' : "glm",
    'time' : -1,
    'citcio': False,
    'add_wy': False,
    'm' : 10
}

args_th = args
args_th['sig_prior'] = .1
args_th['prop_miss'] =  0.3 
args_th['add_wy'] = False 