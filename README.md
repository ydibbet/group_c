# MADS-hackathon

## how to use this
First, go to git and make your own repo. For the naming, use the convention: `MADS-HACK-LASTNAME`
where you replace `LASTNAME` with your own last name.

Then, do the following:

1. `git clone https://github.com/raoulg/mads-hackathon.git` to clone this repo
2. change the remote url to your own repo with `git remote set-url origin MADS-HACK-LASTNAME` where you replace `LASTNAME` with your own last name.
3. install rye with `curl -sSf https://rye.astral.sh/get | bash` and use python 3.11 as default
4. run `rye sync` in your repo
5. You might need to do `sudo apt install clang` on the VM if step 4 fails.

To get the data,
1. `sudo apt install git-lfs`
2. run `git clone https://gitlab.com/rgrouls1/hackathon-data.git`

## The case
The junior datascientist has been trying to create some machine learning models for a dataset.
He has already setup some experiments and basic models, but the dataset is unbalanced and he hasnt been able to get the performance he wants, especially for the classes that are underrepresented.

Because you and your team have been bragging about their machine learning skills, he has asked your team to help him out.

1. Have a look at the dataset, use notebooks 01, 02 and 03 to get a feel for the data and the models the junior has created
2. Brainstorm with your team about approaches to tackle this:
    - what models could you use? Think about all the possible architectures and approaches that might work.
    - What kind of layers could you add to improve the model?
    - what hyperparameters ranges could you tune?
3. Make a plan with your team, and divide the workload. Set your own `dev` tag in the config.toml file, and add the correct port you have received from your teacher. Explore some ideas
4. There is a setup for a hypertune.py file with ray, but it hasnt been implemented for this dataset. Make a plan with your team and start hypertuning.

## The data
### Arrhythmia Dataset

- Number of Samples: about 100k, split into train/valid and a testset you dont have access to
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes:
    - 'Normal': 0,
    - 'Supraventricular ectopic beat': 1,
    - 'Ventricular ectopic beat': 2,
    - 'Fusion beat': 3,
    - 'Unknown beat': 4
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

## Models
There are two notebooks with models. He has tried two approaches: a 2D approach, and a 1D approach. Yes, he is aware that this is a time series, but he has read that 2D approaches can work well for timeseries, and he has tried it out.

He didn't implement RNNs, but maybe you can try that out as well.

Based on your knowledge about hypertuning, you might want to change and expand the models: add more parameters you might want to tune, add additional layers, and if you feel like it you could also create additional models that you think might be promising. Try to balance exploit (improve what is already there) and explore (creativity/curiosity of new things).
Do some manual hypertuing to get a feel for the models and what might work before you start hypertuning with ray.

Analyze the results. It is kind of easy to get high accuracy on the biggest class (normal) but much more difficult to get good results on the other classes.

Because this is a medical dataset, an we are trying to spot disease, in general it is much more important that you find as much sick people as possible, even if that means you will have more false positives.

## The goal
1. Store your best models in the `models` folder.
2. Create a Makefile that runs your best model on the validset. Make sure you set the correct amount of epochs etc, such that the ONLY thing we have to do is run `make run` to get the results.
3. log the results for all your metrics in the `results` folder in `output.csv`. Use the `__repr__` function of the metrics as column names. Add the diagonal of the confusion matrix with the convention `TP_{i}` where `i` is the class.
So you will get something like:
```
modeltag, F1scoremicro, F1scoremacro, TP_0, TP_1, TP_2, TP_3, TP_4
```
as column names, with the metrics as the rows, in the `results/output.csv` file.

## snacks
During the day, we will go to the supermarket to get you some things like fruit, snacks, drinks, etc.
If you have a request, please add it to this [google sheets list](https://docs.google.com/spreadsheets/d/1sFgZfqkkiA2yp6A98k4UhO9t0Q2RD43gtu-hUztzCBY/edit?usp=sharing) with your name, timestamp (just to time when you write down the request) and the request itself.
