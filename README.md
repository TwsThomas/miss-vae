# miss-vae


### Running exp:
$ screen -S exp_7

reserve cpu nb 0 to 23:  
$ taskset -c 0-23 python3 exp_expname.py


### get back the results by scp:

```bash
$ scp tschmitt@drago:/home/tao/tschmitt/miss-vae/results/expname.csv /home/thomas/Documents/miss-vae/results
```