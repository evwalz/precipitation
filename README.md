# precipitation
Forecasting precipitation, 1 day at a time.

## Next Steps
* Validate 2.0 and across folds for 1-2 configs previously used.
* Benchmark `torch.compile()` (on that note, we never used benchmark=True) -> why not both

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
Another dependency, not taken care of yet, is `jsonargparse[signatures]`, let's install it:
```
pip install -U "jsonargparse[signatures]>=4.17.0"
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