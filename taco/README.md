# TACO: Learning Task Decomposition via Temporal Alignment for Control

TACO is a learning from demonstration (LfD) algorithm that can be used to learn simple sub policies from complex demonstrations augmented by a task sketch. Paper:  https://arxiv.org/pdf/1803.01840.pdf

## Dependencies
Tensorflow, MujocoPy Numpy, Scipy, Pandas, Seaborn, Matplotlib.

## Datasets

A small dataset for NavWorld can be found in data/nav/dataset_04.p.

You can also collect your own dataset from these domains.

For NavWorld:
```
	python3 nav_world.py collect [dataset_dir] -n 1000
```
For Dial:
```
	python3 jacopinpad_collect collect [dataset_dir] -n [number of datapoints] -l [sketch_length] --permute
```
For Dial (visual):
```
	python3 jacopinpad_collect collect [dataset_dir] -n [number of datapoints] -l [sketch_length] --img_collect
```
Data should go into the respective domain folder in the data folder. 

## Reproducing:
    Add the taco directory to your ~/.bashrc.
    To run using the example dataset use.
```
	python3 experiment_launch.py reprocuce_taco nav taco dataset_04.p -c taco_nav_base.yaml -n 400
```    
    To run using multiple methods use the .sh scripts for the nav domain.

## Evaluating:
    After training is finished:
```
	python3 evaluation.py reproduce_taco nav 
```    
    This will also plot the results for all the models trained within the test_nav domain. Use the -r flag to render the evaluation to a video (for nav only).





