# precipitation

## Installation

### Conda
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