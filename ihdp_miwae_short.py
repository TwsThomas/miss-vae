import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_miwae
from config_ihdp import args

# set here params 
range_prop_miss = [0.1, 0.3, 0, 0.5]

exp_name = 'ihdp_11.1_11'
# 
output = 'results/'+exp_name+'.csv'
    

print('starting ihdp: ' + exp_name)

args_col = list(set(args.keys()) - set(['set_id']))
args = {k: args[k] for k in args_col}
args['set_id_range'] = range(1,1001)
args['sig_prior'] = 1
args['d_miwae'] = 5
args['n_epochs'] = 100

score_data = pd.DataFrame()
for args['prop_miss'] in range_prop_miss:
    t0 = time.time()
    score = ihdp_miwae(**args)
    args['time'] = int(time.time() - t0)
    score_data = pd.concat((score_data, score))
    print('ihdp with ', args)
    print('........... DONE')
    print('in ', int(args["time"]) , ' s  \n\n')

    score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('IHDP with: '+ exp_name+' succesfully ended.')
print('*'*20)