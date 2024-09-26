# Lab-AI - HPC-Tools

## Introduction 

The goal of this repository is to perform a parrelelization on the training loop of model and see for some figures such as the efficiency or the speed compared the the sequential code.

This repository will be divided in 2 parts. First the "BASELINE" part which provide codes which perform a trainning loop on a BERT model with a Squad data set. And the second part "DISTRIBUTED" which provides the parallelization of the sequential code and some figures about the performances.


## BASELINE

First of all, i want to mention that a great part of the model implementation came from a notebook I have found on the internet : (https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset). I adjust the notebook for our needs. 

So, this part is composed by :
    
    - A notebook which explain in detail all the step I have accomplished to train the model : [BASELINE.ipynb](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/BASELINE.py)

    - A python script that run the training for 2 epochs and evaluate it on a test dataset : [BASELINE.py](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/BASELINE.ipynb)


This python script run a training loop with a a100 gpu executing the command :

```
sbatch run.sh
```

It should generate an output similar to [slurm-8705441.out](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/slurm-8705441.out), but with the name corresponding to the SLURM job ID.

Here is the results I obtained for 

The thaining loop :

[Training Results](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/slurm-8703817.out#L247-L249)

The evaluation : 

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/1d9f1d90a416b68af08f99c09e14a8046a09d4c7/slurm-8703817.out#L5789-L5791
