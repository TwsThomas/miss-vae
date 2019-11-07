import numpy as np
import pandas as pd
import time

from main import exp_miwae
from config import args

# set here params 
range_seed = np.arange(0,10)
range_n = [1000, ]
range_p = [5, 100]
range_prop_miss = [0.1, 0.3, 0]
range_sig_prior = [0.1, 1, 10]
range_n_epochs = [10,]# 400]
# d_miwae = [10, 100]
range_sir = [50, 500]

exp_name = 'exp_07.1_11_B'
# 


print('starting exp: ' + exp_name)
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps','res_tau_resid', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps','mul_res_tau_resid', 'dcor_zhat', 'dcor_zhat_mul']

output = 'results/'+exp_name+'.csv'
l_scores = []

for args['seed'] in range_seed:
    for args['model'] in ["dlvm",]:#"lrmf"]:
        for args['n'] in range_n:
            for args['sig_prior'] in range_sig_prior:
                for args['n_epochs'] in range_n_epochs:
                    for args['prop_miss'] in range_prop_miss:
                        for args['p'] in range_p:  
                            for args['num_samples_zmul'] in range_sir:
                        
                                t0 = time.time()
                                score = exp_miwae(**args)
                                args['time'] = int(time.time() - t0)
                                l_scores.append(np.concatenate((list(args.values()),score)))
                                print('exp with ', args)
                                print('........... DONE')
                                print('in ', int(args["time"]) , ' s  \n\n')

                            score_data = pd.DataFrame(l_scores, columns=list(args.keys()) + l_tau)
                            score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('Exp: '+ exp_name+' succesfully ended.')
print('*'*20)