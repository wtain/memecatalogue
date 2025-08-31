# memecatalogue
Meme Organizing catalogue, powered with AI

# Setup

## Create virtual environment
```bash
python3 -m venv ./myenv
```

## Activate virtual environment
```bash
source ./myenv/bin/activate
```

## Install requirements
```bash
pip install -r requirements.txt
```


### MacOS

```commandline
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

### Alternative

```bash
pip install torch torchvision pillow faiss-cpu ftfy regex tqdm git+https://github.com/openai/CLIP.git

pip3 install torch torchvision torchaudio
```

### Anaconda

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

```bash
# Replace <PATH-TO-CONDA> with the path to your conda installation
<PATH-TO-CONDA>\Scripts\activate.bat
```
E.g.:
```bash
C:\Users\ramiz\miniconda3\Scripts\activate.bat
```

```bash
conda create -n MemeCatalogue python=3.9 -y
conda activate MemeCatalogue
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install conda-forge::openai-clip
conda install conda-forge::faiss
```

### Remarks

1. Clip comes from pip install git+https://github.com/openai/CLIP.git
2. 

## Update requirements
```bash
pip3 freeze > requirements.txt
```