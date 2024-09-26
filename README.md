# Lab-AI - HPC-Tools

## Introduction 

The goal of this repository is to perform a parrelelization on the training loop of model and see for some figures such as the efficiency or the speed compared the the sequential code.

This repository will be divided in 2 parts. First the "BASELINE" part which provide codes which perform a trainning loop on a BERT model with a Squad data set. And the second part "DISTRIBUTED" which provides the parallelization of the sequential code and some figures about the performances.


## BASELINE

First of all, i want to mention that a great part of the model implementation came from a notebook I have found on the internet : (https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset). I adjust the notebook for our needs. 

So, this part is composed by :
    
- A notebook which explain in detail all the step I have accomplished to train the model : [BASELINE.ipynb](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/4043e5e602096310e59887368f08188fafe21cfa/BASELINE.ipynb)

- A python script that run the training for 2 epochs and evaluate it on a test dataset : [BASELINE.py](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/4043e5e602096310e59887368f08188fafe21cfa/BASELINE.py)


This python script run a training loop with a a100 gpu executing the command :

```
sbatch run.sh
```

It should generate an output similar to [slurm-8705441.out](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/slurm-8705441.out), but with the name corresponding to the SLURM job ID.

Here is the results I obtained for 

The thaining loop :

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/slurm-8703817.out#L247-L249

The evaluation : 

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/1d9f1d90a416b68af08f99c09e14a8046a09d4c7/slurm-8703817.out#L5789-L5791

So we can see that the total training time of the model with the whole dataset for 2 epoch is about 1 hour eand the loss fonction is very small at the end (0.88)

The loss fonction of the validation dataset is also very small (0.55), so thhe model seems very performant.

Finally, you can see the evolution of the loss function in both training and evaluation with tensorboard by executing this command : 

```
tensorboard --logdir=logs/fit
```

Here are the results I optained on tensorbaord with my try : 


ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

# Lab-AI - HPC-Tools

## Introduction 

This repository focuses on parallelizing the training loop of a machine learning model and analyzing key metrics such as efficiency and speedup in comparison to the sequential implementation.

The repository is divided into two parts:

1. **BASELINE**: A sequential implementation of a training loop on a BERT model using the SQuAD dataset.
2. **DISTRIBUTED**: The parallelized version of the code, along with performance metrics and comparisons.

## BASELINE

A significant part of the model implementation in this section is based on a notebook found online: [BERT-based pretrained model using SQuAD 2.0 dataset](https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset). I have adapted the notebook to suit the objectives of this project.

This section includes the following:

- A detailed notebook explaining the steps taken to train the model: [BASELINE.ipynb](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/BASELINE.ipynb).
- A Python script that trains the model for 2 epochs and evaluates it on a test dataset: [BASELINE.py](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/BASELINE.py).

To run the training script on an A100 GPU, you can use the following command:

```
sbatch run.sh
```


This will generate an output similar to this SLURM output file: [slurm-8705441.out](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/slurm-8705441.out), but the file name will match the corresponding SLURM job ID.

### Results

You can find the results of the training loop here:

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/693edd42dcd61574d18b0b798045b69d592c26b6/slurm-8703817.out#L247-L249

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/1d9f1d90a416b68af08f99c09e14a8046a09d4c7/slurm-8703817.out#L5789-L5791

The total training time for 2 epochs on the full dataset is approximately 1 hour, with a final training loss of **0.88**.

The validation loss is also very low at **0.55**, indicating strong model performance.

#### Visualization

You can visualize the evolution of the training and validation loss using TensorBoard by running:

```
tensorboard --logdir=logs/fit
```

Below is a snapshot of the results obtained from my experiments:


