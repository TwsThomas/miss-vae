import numpy as np
import pandas as pd
import time

from main_ihdp import ihdp_miwae_cv
from config_ihdp import args

# set here params 
range_prop_miss = [0.5,]#[0.1, 0.3, 0]
range_sig_prior = [0.1, 1, 10]
range_n_epochs = [100,] #[10,]# 400]
range_d_miwae = [5, 10, 100]

exp_name = 'ihdp_cv_10.2_11'
# 


print('starting ihdp: ' + exp_name)

l_elbo = ['d_miwae', 'sig_prior', 'k', 'elbo']

output = 'results/'+exp_name+'.csv'
l_scores = []

args_col = list(set(args.keys()) - set(['sig_prior','d_miwae']))
args = {k: args[k] for k in args_col}
args['k_fold'] = 5

for args['set_id'] in range(1,1001, 100):
    for args['prop_miss'] in range_prop_miss:
        for args['add_wy'] in [False]:#, True]:
            for args['n_epochs'] in range_n_epochs:
                t0 = time.time()
                score = ihdp_miwae_cv(**args, sig_prior_list = range_sig_prior, d_miwae_list = range_d_miwae)
                args['time'] = int(time.time() - t0)
                for i in range(len(score)):
                    l_scores.append(np.concatenate((list(args.values()),score[i])))
                print('ihdp with ', args)
                print('........... DONE')
                print('in ', int(args["time"]) , ' s  \n\n')

            score_data = pd.DataFrame(l_scores, columns=list(args.keys()) + l_elbo)
            score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('IHDP with: '+ exp_name+' succesfully ended.')
print('*'*20)