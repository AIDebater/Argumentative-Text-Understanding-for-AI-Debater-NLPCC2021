## Requirements
See requirement.txt or Pipfile for details
* pytorch==1.7.1
* transformers==3.4.0
* python=3.6

## Usage
- ### Training
For example, you can use the folowing command to train the model:
```
CUDA_VISIBLE_DEVICES=0 python main.py --task pair --mode train --dataset rr-submission-v2 --batch_size 1 --epoch 25 --lr 2e-4 --pair_weight 0.5 --cls_method binary --class_num 2 --model BaselineLSTMModel --model_dir your_model/ --token_embedding True --freeze_bert True --optimizer adam --max_bert_token 100
```
The best model will be saved in the folder "your_model/".

- ### Testing
For example, you can use the folowing command to test the model:
```
CUDA_VISIBLE_DEVICES=0 python main.py --task pair --mode test --dataset rr-submission-v2 --batch_size 1 --epoch 25 --lr 2e-4 --pair_weight 0.5 --cls_method binary --class_num 2 --model BaselineLSTMModel --model_dir your_model/ --token_embedding True --freeze_bert True --optimizer adam --max_bert_token 100
```

