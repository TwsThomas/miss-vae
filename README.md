# miss-vae


### Running exp:  
$ screen -S exp_7

reserve cpu nb 0 to 23:  
$ taskset -c 0-23 python3 exp_expname.py


### get back the results by scp:

```bash
$ scp tschmitt@drago:/home/tao/tschmitt/miss-vae/results/expname.csv /home/thomas/Documents/miss-vae/results
```


### Running cevae model :  

```bash
conda config --append channels conda-forge
conda create -n cevae_env numpy pandas scikit-learn==0.18.1  tensorflow==1.1.0 progressbar==2.3 pip scipy tensorflow-probability
conda activate cevae_env
pip install edward==1.3.1
```