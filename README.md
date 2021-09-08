# td-opswtw-competition-rl
2nd place solution to the [AI for TSP](https://www.tspcompetition.com) ([github](https://github.com/paulorocosta/ai-for-tsp-competition)) competition.

This solution is an adaptation of Pointer Network model in:

*[A reinforcement learning approach to the orienteering problem with time windows](https://www.sciencedirect.com/science/article/pii/S0305054821001349) <br/>
[Gama R](https://scholar.google.com/citations?hl=en&user=uHKwsF0AAAAJ&view_op=list_works&sortby=pubdate), 
[Fernandes HL](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=JG7xb2AAAAAJ&sortby=pubdate) - Computers & Operations Research, 2021* 
  ([arXiv](https://arxiv.org/abs/2011.03647), [github](https://github.com/mustelideos/optw_rl))



## 1. Setup environment
1.1. Install conda env environment_plus.yml
    
```console
$ conda env create --file environment_plus.yml
```
    
Same as the competition environment but with some added a packages: 
      `tqdm`, `tensorboard`, `matplotlib`, `ipykernel`
    
1.2. Activate environmennt
```console
$ conda activate tsp-ai-competition-plus
```

1.3. Install a jupyter notebook kernel for `tsp-ai-competition-plus` conda env
```bash
python -m ipykernel install --name tsp-ai-competition-plus
```


## 2. Pre-generate training instances and compute normalization stats

2.1. Generate training instances
```bash
python -m generator.op.instances_plus
```

This will pre-generate instances with `seed` from `1` to `4000` for `n_nodes` `10` to `210`. These are randomly sampled during training.
For the submitted model we only train on `n_nodes` `10` to `125`.
    
2.2. Compute andd save normalization statistics for static features

Run notebook: `/notebooks/1.0.0-rg-instances-stats.ipynb`

Make sure the notebook kernel is set to `tsp-ai-competition-plus`

## 3. Train and generate submission:

3.1. Unzip validation and test instances in `/data/valid` and `data/test`

3.2. Run notebook:
`/notebooks/2.0.0-rg-optw-paper-model`

Make sure the notebook kernel is set to `tsp-ai-competition-plus`
