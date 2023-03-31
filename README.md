# precipitation

## Installation (outdated - see below)
Creating the environment.
```
conda env create --name precip python=3.10
```
conda activate precip
Cartoy requires GEOS, Shapely and pyshp - it's easiest to conda install.
```
conda install -c conda-forge cartopy
```
Also opted to conda-install pytorch with the latest CUDA 11.6 version. Note: This requires driver versions >450.80.02. If no CUDA-capable GPU is available, probably you'll need to `pip install torch==1.12.0`.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
Then you should be fine to install the remaining requirements using pip.
```
pip install -r requirements.txt
```


## Step-by-Step Install

First, create the env: (2.0 refers to major updates to Lightning and PyTorch):
```
conda create -n precip_2.0 python=3.10 
```
Activate the environment:
```
conda activate recip_2.0
```
First, the long PyTorch install (Using Cuda 11.x requires driver version >450.80.02):
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Now for Lighting:
```
pip install lightning
```
Apparently, pytorch lightning now checks if torch is present (even from conda installs), and doesn't try to reinstall pytorch. Great! Afaik, this used to be the case before.

Since cartopy (used in this project) requires GEOS, Shapely and pyshp - let's also conda install this:
```
conda install -c conda-forge cartopy
```

Now we can pip install the remainind dependencies from the requirements.txt:
```
pip install -r requirements.txt
```

Now we need Eva's isodisreg package. Go some point where you want to clone that other repo, and clone + install:
```
cd ~/precipitation/repos
git clone https://github.com/evwalz/isodisreg.git
cd isodisreg
pip install -r requirements.txt
pip install -e .
```

Finally, let's install what we're after (we already installed the requirements above):
```
cd ~/precipitation/repos/
git clone https://github.com/evwalz/precipitation.git
pip install -e .
```