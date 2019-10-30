# miss-vae


### Running exp:  
$ screen -S exp_7

reserve cpu nb 0 to 23 and set the niceness (for server):
$ taskset -c 0-23 nice -5 python exp_expname.py


### get back the results by scp:

```bash
$ scp tschmitt@drago:/home/tao/tschmitt/miss-vae/results/expname.csv /home/thomas/Documents/miss-vae/results
```


### Running cevae model :  

```bash

conda config --append channels anaconda 
conda config --append channels conda-forge
conda create -n cevae_env python numpy pandas joblib scikit-learn==0.18.1  tensorflow==1.1.0 progressbar==2.3 pip scipy 
conda activate cevae_env
pip install edward==1.3.1 --user

taskset -c 0-23 nice -5 python2 exp_cevae_name.py
```

#### Other issues:
pip3 issues, had to re-install : 
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py --force-reinstall

