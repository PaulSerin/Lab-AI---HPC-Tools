# Lab-AI - HPC-Tools

## Introduction 

The goal of this repository is to perform a parrelelization on the training loop of model and see for some figures such as the efficiency or the speed compared the the sequential code.

This repository will be divided in 2 parts. First the "BASELINE" part which provide codes which perform a trainning loop on a BERT model with a Squad data set. And the second part "DISTRIBUTED" which provides the parallelization of the sequential code and some figures about the performances.


## BASELINE

First of all, i want to mention that a great part of the model implementation came from a notebook I have found on the internet : (https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset). I adjust the notebook for our needs. 

So, this part is composed by :
    
    - A notebook which explain in detail all the step I have accomplished to train the model : [BASELINE.ipynb](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/BASELINE.ipynb)

    - A python script that run the training for 2 epochs and evaluate it on a test dataset : [BASELINE.py](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/BASELINE.py)


This python script run a training loop with a a100 gpu executing the command :

```
sbatch run.sh
```

It should generate an output similar to [slurm-8703817.out](https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/slurm-8703817.out), but with the name corresponding to the SLURM job ID.

Here is the results I obtained for the thaining loop : 

https://github.com/PaulSerin/Lab-AI---HPC-Tools/blob/main/slurm-8700021.out#L30-L37




https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L10

which provides implementations for several methods:
- The constructor *__init__*, which contains the model definition.
- The implementation of the *forward* pass.
- The method to configure the optimizer(s): *configure_optimizers*.
- The definition of a *training_step*. Let's see its content in the code:
  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L30-L37

- The definition of a *validation_step*. Let's see its content in the code:

  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L39-L45

  By providing implementations for all these methods, using Lightning is really simple and involves just a few lines of code:

  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L48-L60

  Additionally, we see that Lightning can automatically detect and manage the *world_size* and *rank* provided by SLURM.

# Ejemplo inicial de Lightning

Este código contiene un ejemplo básico de cómo lanzar un entrenamiento distribuido en Lightning en un entorno distribuido gestionado con SLURM.

El ejemplo se lanza ejecutando el comando

```
sbatch sampleLI.sbatch
```
y debería generar una salida similar a [slurm-3417863.out](https://github.com/diegoandradecanosa/CFR24/blob/main/pytorch_dist/lightning/000/slurm-3417863.out)
pero con el nombre correspondiente al id del trabajo SLURM.

El script de entrenamiento es el fichero [sampleLI.py](https://github.com/diegoandradecanosa/CFR24/blob/main/pytorch_dist/lightning/000/sampleLI.py)

En el script de entrenamiento la definición del modelo se debe hacer dentro de un *LightningModule*

https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L10

a través del cual se proporciona una implementación para varios métodos:
- El constructor *__init__* que contiene la definición del modelo
- La implementación de la pasada *forward*
- El método para configurar el/los optimizadores *configure_optimizers*
- La definición de un *training_step*. Veamos su contenido en el código.
  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L30-L37
  
- La definición de un *validation_step*. Veamos su contenido en el código.

  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L39-L45

  Si proporcionamos una implementación para todos esos métodos, el uso de Lightning es realmente sencillo e involucra unas pocas líneas de código

  https://github.com/diegoandradecanosa/CFR24/blob/40f1fbc4602610e2e2fe60f760f5b2d1bb99ba30/pytorch_dist/lightning/000/sampleLI.py#L48-L60

  Además, vemos que lightning es capaz de detectar y gestionar automáticamente el *world_size* y el *rank* proporcionados por SLURM.
  