# BIT AI

This repo contains Jupyter Notebooks and presentations presented during the meetings of the AI section of Bit Scientific Group @ Cracow University of Science and Technology, where we dive deep into Machine Learning awesomeness.

The notebooks are intended to compose an introductory Machine Learning course, tackling topics ranging from linear regression to various forms of Deep Learning. Their aim is to introduce the reader to the math and techniques behind ML, as well as the most popular programming tools designed with ML in mind, such as Scikit-Learn and PyTorch.

## Setup

All notebooks are written and tested in Python 3.6.6 and an Unix OS.  Their usage is not guaranteed (though probably possible) with other versions of the language and other OS's.

We recommend [Anaconda](https://www.anaconda.com/download/#linux) for managing packages and Python environments.

1. [Download, install Anaconda and add it to your `$PATH` environment variable.](http://docs.anaconda.com/anaconda/install/linux/) 

2. Create a new Python environment for your BIT AI excercises

```bash
conda create -n bit_ai -- python=3.6.6 jupyter
```

3. Clone the repository and enter the cloned directory

```bash
git clone https://github.com/aghbit/BIT_AI.git
cd BIT_AI
```

4. Activate the environment and install required packages

```bash
source activate bit_ai
pip install -r requrements.txt
```

That's it!

## Running the notebooks

Before each meeting of BIT AI, it is always advised to update the repo - you never know what last-minute additions may have been made! 

```bash
cd BIT_AI
git pull
```

Remember to acivate your Conda Environment before running the notebooks!

```bash
source activate bit_ai
jupyter-notebook
```