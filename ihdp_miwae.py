import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_miwae
from config_ihdp import args

# set here params 
range_prop_miss = [0.1, 0.3, 0]
range_sig_prior = [0.1, 1, 10]
range_n_epochs = [10, 400]
range_d_miwae = [10, 100]

exp_name = 'ihdp_04.1_11'
# 


print('starting ihdp: ' + exp_name)

l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps','tau_resid', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps','mul_tau_resid', 'dcor_zhat', 'dcor_zhat_mul']

output = 'results/'+exp_name+'.csv'
l_scores = []

for args['set_id'] in range(1,1001):
    for args['prop_miss'] in range_prop_miss:
        for args['add_wy'] in [False]:#, True]:
            for args['sig_prior'] in range_sig_prior:
                for args['n_epochs'] in range_n_epochs:
                    for args['d_miwae'] in range_d_miwae:     
                                
                        t0 = time.time()
                        score = ihdp_miwae(**args)
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