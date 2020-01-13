# archive file. Only mean for historicall archive.

def plot_n_d():
    range_seed = np.arange(10)
    range_n = [10**4, 10**6]
    range_p = [20, 100, 1000]
    range_prop_miss = [0.1, 0.3, 0]
    range_sig_prior = [0.1, 1, 10]

    l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']
    args_name = ['model', 'seed', 'n', 'p', 'prop_miss', 'sig_prior']
    exp_name = 'plot_nd'

    l_scores = []
    for model in ["dlvm","lrmf"]:
        for seed in range_seed:
            for prop_miss in range_prop_miss:
                for n in range_n:
                    for p in range_p:     
                        for sig_prior in range_sig_prior:
                            # print('start with', n, p, prop_miss)
                            score = exp(model = model, n=n, d=3, p=p, prop_miss=prop_miss,
                                        seed=seed, d_miwae=3,
                                        sig_prior=sig_prior, n_epochs=602)
                            args = (model, seed, n, p, prop_miss, sig_prior)
                            l_scores.append(np.concatenate((args,score)))
                            print('exp with ', args_name)
                            print(args)
                            print('........... DONE !\n\n')

                score_data = pd.DataFrame(l_scores, columns=args_name + l_tau)
                score_data.to_csv('results/'+exp_name+'_temp.csv')
    
    print('saving ' +exp_name + 'at: results/'+exp_name+'.csv')
    score_data.to_csv('results/'+exp_name+'.csv')
    
    
def plot_epoch():

    l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']
    args_name = ['model','n', 'n_epochs']

    l_scores = []
    for model in ["dlvm"]:
        for n in [200, 1000, 10000]:
            for n_epochs in [10, 100, 400, 600, 800]:
                score = exp(model = model, n=n, d=3, p=100, prop_miss=0.1, seed = 0,
                    d_miwae=3, n_epochs=n_epochs, sig_prior = 1,
                    method = "glm")
                args = (model ,n, n_epochs)
                l_scores.append(np.concatenate((args,score)))
                print('exp with ', args_name)
                print(args)
                print('........... DONE !\n\n')

                score_data = pd.DataFrame(l_scores, columns=args_name + l_tau)
                score_data.to_csv('results/plot_epoch_temp.csv')
    
    score_data.to_csv('results/plot_epoch.csv')