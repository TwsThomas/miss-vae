# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_cevae
from config_ihdp import args

# set here params 
range_prop_miss = [0.1, 0.3, 0]

exp_name = 'ihdp_05.1_11_cevae'
# 


print('starting ihdp: ' + exp_name)

l_tau = ['tau_cevae']

output = 'results/'+exp_name+'.csv'
l_scores = []
args['n_epochs'] = 202

for args['set_id'] in range(1,1001):
    for args['prop_miss'] in range_prop_miss:
                                
        t0 = time.time()
        score = ihdp_cevae(**args)
        args['time'] = int(time.time() - t0)
        l_scores.append(np.concatenate((list(args.values()),score)))
        print('ihdp with ', args)
        print('........... DONE')
        print('in ', int(args["time"]) , ' s  \n\n')

    score_data = pd.DataFrame(l_scores, columns=list(args.keys()) + l_tau)
    score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('IHDP with: '+ exp_name+' succesfully ended.')
print('*'*20)