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

### Alternative

```bash
pip install torch torchvision pillow faiss-cpu ftfy regex tqdm git+https://github.com/openai/CLIP.git

```

### Remarks

1. Clip comes from pip install git+https://github.com/openai/CLIP.git
2. 

## Update requirements
```bash
pip3 freeze > requirements.txt
```