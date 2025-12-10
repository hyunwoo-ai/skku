### Step 1

```
cd work
conda deactivate
python -m venv skku_venv
source skku_venv/bin/activate
```

### Step 2

```
pip install -r skku/requirements.txt
```

### Step 3

```
python3 -m ipykernel install --user --name skku_venv --display-name "skku_venv"
```
