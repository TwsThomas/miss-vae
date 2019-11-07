import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_mi
from config_ihdp import args

# set here params 
range_prop_miss = [0.5, 0.7, 0.9] #[0.1, 0.3, 0]
#range_sig_prior = [0.1, 1, 10]
#range_n_epochs = [10, 200, 600]

exp_name = 'ihdp_mi_2'
# 

args['m'] = 10
print('starting exp: ' + exp_name)
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps']
output = 'results/2019-11-07_'+exp_name+'.csv'
l_scores = []

for args['set_id'] in range(1,1001):
    for args['prop_miss'] in range_prop_miss:
        t0 = time.time()
        score = ihdp_mi(**args)
        args['time'] = int(time.time() - t0)
        l_scores.append(np.concatenate((list(args.values()),score)))
        print('ihdp_mi with ', args)
        print('........... DONE')
        print('in ', int(args["time"]) , ' s  \n\n')

    score_data = pd.DataFrame(l_scores, columns=list(args.keys()) + l_tau)
    score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('IHDP with: '+ exp_name+' succesfully ended.')
print('*'*20)