# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_cevae
from config_ihdp import args

# set here params 

exp_name = 'ihdp_30_10_cevae'
# 


print('starting ihdp: ' + exp_name)

l_tau = ['tau_cevae']

output = 'results/'+exp_name+'.csv'
l_scores = []

for args['set_id'] in range(1,11):
    for args['n'] in [1000, 5000]:
        for args['citcio'] in [False, True]:
            for args['n_epochs'] in [10,402]:
                for args['p'] in [5, 100]:
                    for args['prop_miss'] in [0, 0.1, 0.3]:
                        for args['model'] in ["dlvm", "lrmf"]:
                            t0 = time.time()
                            score = [ihdp_cevae(**args)]
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