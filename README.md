# deblur
deblur model for Takeme2space Project

## Setting up
1. install [Anaconda](https://www.anaconda.com/download/success)
2. Run "Anaconda Prompt" terminal and run the following code
```
conda create -n "takeme2space" python=3.9.7
``` 
3. activate takeme2space enviroment
```
conda activate takeme2space
```
4. upgrade pip
```
python -m pip install --upgrade pip
```
5. install required packages
```
pip install -r requirements.txt
```
6. Download and unzip [dataset](https://sotonac-my.sharepoint.com/:u:/g/personal/fsm1d23_soton_ac_uk/EebAPMYDCwhJkJ68o4sOkZsB4SBphhcKtKhxaldugsDlnA?e=OiFmP0)


## Usage
### train model
```
python deblur_tf.py -t
```

### convert model from tensorflow to tensorflow lite
```
python deblur_tf.py -c
```

### inference/testing image
```
python deblur_tf.py -i
```
   * store blurry images in "test_image" folder
   * results will be saved in "result" folder

### Change path to save model
```
python deblur_tf.py -p train1
```
   * this example will save the models to a folder called "train1"
   * if the flag -p is not used, the default folder is saved_model

### Train --> Convert --> inference
```
python deblur_tf.py -t -c -i
```

# Development Guide

### Creating new branch
```
git branch dev-YOURNAME
```

### Switching to new branch
```
git checkout branch dev-YOURNAME
```

### checking status
```
git status
```
- to check status of files, red represents files that are modified and not yet pushed

### add files before commiting
```
git add fileName1, fileName2, fileName3
```

### commit files
```
git commit -m "any message for reference purposes"
```

### push files to git repo
```
git push -u origin dev-YOURNAME
```
### Note: Ensure to push to your branch only
