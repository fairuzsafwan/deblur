# deblur
deblur model for Takeme2space Project

## Setting up
- python -m pip install --upgrade pip
- pip install -r requirements.txt

## Usage
### train model
```
python deblur_tf.py -t
```

### convert model from tensorflow to tensorflow lite
```python deblur_tf.py -c
```

### inference/testing image
```python deblur_tf.py -i
```
   * ensure to store blurry images in "test_image" folder
   * results will be saved in "result" folder

### Change path to save model
```python deblur_tf.py -p train1
```
   * this example will saved the models to a folder called "train1"
   * if the flag -p is not used, the default folder is saved_model

### Train --> Convert --> inference
```python deblur_tf.py -t -c -i
```
