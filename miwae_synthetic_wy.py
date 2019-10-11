# !pip3 install --user --upgrade scikit-learn # We need to update it to run missForest
import tensorflow as tf
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import os

def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])



## modified /
def run_main(id_data = 0, setting = "dlvm2", link = "nonlinear3", 
             n = 1000, p = 10, d=3, d_miwae = 3, h_miwae=128, add_mask=True,
             num_samples_zmul=200, num_samples_xmul=50, perc_miss = 0.1,
             in_folder = "../Data/",
             out_folder = "../Data/"):
  ##########
  id = id_data
  print(in_folder, out_folder)
  print('run_main', id_data, n, p)
  
  
  #
  try:
      os.makedirs(in_folder, exist_ok=True)
  except FileExistsError:
      pass

  try:
      os.makedirs(out_folder, exist_ok=True)
  except FileExistsError:
      pass
  
  in_file_text = setting + "_" + link + "_" + str(n) + "_" + str(p) + "_" + str(d) + "_seed" + str(id) + "_propNA"+"{:3.2f}".format(perc_miss)
  if add_mask:
    out_file_text = setting + "_" + link + "_" + str(n) + "_" + str(p) + "_" + str(d) + "_WY_dmiwae" + str(d_miwae) + "_hmiwae" + str(h_miwae) + "_mask_seed" + str(id) + "_propNA"+"{:3.2f}".format(perc_miss)
  else:
    out_file_text = setting + "_" + link + "_" + str(n) + "_" + str(p) + "_" + str(d) + "_WY_dmiwae" + str(d_miwae) + "_hmiwae" + str(h_miwae) + "_seed" + str(id) + "_propNA"+"{:3.2f}".format(perc_miss)
  
  print(out_folder+out_file_text)

  data = np.array(pd.read_csv(in_folder+in_file_text+"_xcomp.csv", low_memory=False))[:,1:(p+1)]
  data_miss = np.array(pd.read_csv(in_folder+in_file_text+"_xmisswy.csv", low_memory=False))[:,1:(p+3)]
  
  print('dim(data)', data.shape)
  print('dim([data_miss , w])', data_miss.shape)

  # #########

  xfull = (data - np.mean(data,0))/np.std(data,0)
  n = xfull.shape[0] # number of observations
  p = xfull.shape[1] # number of features

  print(n)
  print(p)

  # ##########

  np.random.seed(1234)
  tf.set_random_seed(1234)

  xmiss = np.copy(data_miss)
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing

  print('mask', mask.shape)
  
  # ##########

  xhat_0 = np.copy(xmiss)
  xhat_0[np.isnan(xmiss)] = 0
  
  p_mod = p
  pw = 2
  if add_mask:
    mask_mod = np.copy(mask)
    #mask_mod = mask_mod.astype(float)
    #mask_mod = (mask_mod - np.mean(mask_mod,0))/np.std(mask_mod,0)
    xhat_0 = np.concatenate((xhat_0, mask_mod), axis=1)
    xfull = np.concatenate((xfull, mask_mod), axis=1)
    mask = np.concatenate((mask, np.ones_like(mask).astype(bool)), axis = 1)
    p = p*2
    print('[data, w, y, mask, 1, 1]', xhat_0.shape)
    pw = 4

  

  # ##########

  x = tf.placeholder(tf.float32, shape=[None, p+pw]) # Placeholder for xhat_0
  learning_rate = tf.placeholder(tf.float32, shape=[])
  batch_size = tf.shape(x)[0]
  xmask = tf.placeholder(tf.bool, shape=[None, p+pw])
  K= tf.placeholder(tf.int32, shape=[]) # Placeholder for the number of importance weights

  # ##########

  

  p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(d_miwae, tf.float32))

  # ##########

  sigma = "relu"
  
  decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[d_miwae,]),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(3*(p+pw),kernel_initializer="orthogonal") # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
  ])

  # ##########

  tiledmask = tf.tile(xmask,[K,1])
  tiledmask_float = tf.cast(tiledmask,tf.float32)
  mask_not_float = tf.abs(-tf.cast(xmask,tf.float32))

  iota = tf.Variable(np.zeros([1,p+pw]),dtype=tf.float32)
  tilediota = tf.tile(iota,[batch_size,1])
  iotax = x + tf.multiply(tilediota,mask_not_float)

  # ##########

  encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[p+pw,]),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(h_miwae, activation=sigma,kernel_initializer="orthogonal"),
    tfkl.Dense(3*d_miwae,kernel_initializer="orthogonal")
  ])

  # ##########

  out_encoder = encoder(iotax)
  q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d_miwae], scale=tf.nn.softplus(out_encoder[..., d_miwae:(2*d_miwae)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d_miwae):(3*d_miwae)])))
  zgivenx = q_zgivenxobs.sample(K)
  zgivenx_flat = tf.reshape(zgivenx,[K*batch_size,d_miwae])
  data_flat = tf.reshape(tf.tile(x,[K,1]),[-1,1])

  # ##########

  out_decoder = decoder(zgivenx_flat)
  all_means_obs_model = out_decoder[..., :(p+pw)]
  all_scales_obs_model = tf.nn.softplus(out_decoder[..., (p+pw):(2*(p+pw))]) + 0.001
  all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*(p+pw)):(3*(p+pw))]) + 3
  all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
  all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[K*batch_size,p+pw])

  # ##########

  logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz[:,0:p_mod],tiledmask_float[:,0:p_mod]),1),[K,batch_size])
  logpz = p_z.log_prob(zgivenx)
  logq = q_zgivenxobs.log_prob(zgivenx)

  # ##########

  miwae_loss = -tf.reduce_mean(tf.reduce_logsumexp(logpxobsgivenz + logpz - logq,0)) +tf.log(tf.cast(K,tf.float32))
  train_miss = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(miwae_loss)

  # ##########

  xgivenz = tfd.Independent(
        distribution=tfd.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model))

  # ##########

  imp_weights = tf.nn.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
  xms = tf.reshape(xgivenz.mean(),[K,batch_size,p+pw])
  xm=tf.einsum('ki,kij->ij', imp_weights, xms) 

  # ##########

  z_hat = tf.einsum('ki,kij->ij', imp_weights, zgivenx) 

  # ##########

  sir_logits = tf.transpose(logpxobsgivenz + logpz - logq)
  sirx = tfd.Categorical(logits = sir_logits).sample(num_samples_xmul)
  xmul = tf.reshape(xgivenz.sample(),[K,batch_size,p+pw])

  sirz = tfd.Categorical(logits = sir_logits).sample(num_samples_zmul)
  zmul = tf.reshape(zgivenx,[K,batch_size,d_miwae])
  
  # ##########

  miwae_loss_train=np.array([])
  mse_train=np.array([])
  bs = 64 # batch size
  n_epochs = 602
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
  x_mul_imp = np.tile(xhat_0,[num_samples_xmul,1,1])
  zhat = np.zeros([n,d_miwae]) # low-dimensional representations

  zhat_mul = np.tile(zhat, [num_samples_zmul,1,1])
      
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(1,n_epochs):
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask[perm,], n/bs)
      for it in range(len(batches_data)):
          train_miss.run(feed_dict={x: batches_data[it], learning_rate: 0.001, K:20, xmask: batches_mask[it]}) # Gradient step      
      if ep % 200 == 1:
        losstrain = np.array([miwae_loss.eval(feed_dict={x: xhat_0, K:20, xmask: mask})]) # MIWAE bound evaluation
        miwae_loss_train = np.append(miwae_loss_train,-losstrain,axis=0)
        print('Epoch %g' %ep)
        print('MIWAE likelihood bound  %g' %-losstrain)
        for i in range(n): # We impute the observations one at a time for memory reasons
            # # Single imputation:
            xhat[i,:][~mask[i,:]]=xm.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pw]), K:10000, xmask: mask[i,:].reshape([1,p+pw])})[~mask[i,:].reshape([1,p+pw])]
            # # Multiple imputation:
            # si, xmu = sess.run([sirx, xmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pw]), K:10000, xmask: mask[i,:].reshape([1,p+pw])})
            # x_mul_imp[:,i,:][~np.tile(mask[i,:].reshape([1,p+pw]),[num_samples_xmul,1])] = np.squeeze(xmu[si,:,:])[~np.tile(mask[i,:].reshape([1,p+pw]),[num_samples_xmul,1])]
            # Dimension reduction:
            zhat[i,:] = z_hat.eval(feed_dict={x: xhat_0[i,:].reshape([1,p+pw]), K:10000, xmask: mask[i,:].reshape([1,p+pw])})
            # Z|X* sampling:
            si, zmu = sess.run([sirz, zmul],feed_dict={x: xhat_0[i,:].reshape([1,p+pw]), K:10000, xmask: mask[i,:].reshape([1,p+pw])})    
            zhat_mul[:,i,:] = np.squeeze(zmu[si,:,:])
        err = np.array([mse(xhat[:,:p_mod],xfull[:,:p_mod],mask[:,:p_mod])])
        mse_train = np.append(mse_train,err,axis=0)
        print('Imputation MSE  %g' %err)
        print('-----')

  # ##########

  
  print('save files')
  if add_mask:
    xhat_rescaled = xhat[:,0:int(p/2)]*np.std(data,0) + np.mean(data,0)
  else:
    xhat_rescaled = xhat[:,:p]*np.std(data,0) + np.mean(data,0)
  np.savetxt(out_folder+out_file_text+"_imp.csv", xhat_rescaled, delimiter=";")
    
  
  # xmiss_rescaled = xmiss*np.std(data,0) + np.mean(data,0)
  # np.savetxt(out_folder+"/"+out_file_text+"_xmiss.csv", xmiss_rescaled, delimiter=";")
  
  np.savetxt(out_folder+out_file_text+"_zhat.csv", zhat, delimiter=";")

  
  # if add_mask:
  #   x_mul_imp_rescaled = x_mul_imp[:,:,0:int(p/2)]
  # else:
  #   x_mul_imp_rescaled = x_mul_imp
  # for i in range(num_samples_xmul):
  #   x_mul_imp_rescaled[i,:,:p] = x_mul_imp_rescaled[i,:,:p]*np.std(data,0) + np.mean(data,0)
  #   np.savetxt(out_folder+"/"+out_file_text+"_imp_m"+str(i)+".csv", x_mul_imp_rescaled[i,:,:p], delimiter=";")

    
  zhat_mul_rescaled = zhat_mul
  for i in range(num_samples_zmul):
    np.savetxt(out_folder+out_file_text+"_zhat_m"+str(i)+".csv", zhat_mul_rescaled[i,:,:], delimiter=";")

