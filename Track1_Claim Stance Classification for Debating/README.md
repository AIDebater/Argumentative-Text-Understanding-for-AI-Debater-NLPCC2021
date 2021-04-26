# Track 1: Claim Stance Classification for Debating
This project serves as the baseline model for **[Argumentative-Text-Understanding-for-AI-Debater-NLPCC2021](https://github.com/AIDebater/Argumentative-Text-Understanding-for-AI-Debater-NLPCC2021) Track 1: Claim Stance Classification for Debating**.


## 0. Set Up
### 0.1 Dataset
This project uses `train.txt` and `test.txt` provided in the first phase of the competition. Please refer to the [official website](http://www.fudan-disc.com/sharedtask/AIDebater21/index.html) for competition registration and dataset downloading.

The dataset used in this project contains 6,416 training instances and 990 test instances. Each line contains three fields: `TOPIC<tab>CLAIM<tab>STANCE LABEL`. Note that the topics in the training and test sets are not overlapped. 


### 0.2 Requirements
- Python >= 3.6 and PyTorch >= 1.6
- simpletransformers [link](https://github.com/ThilinaRajapakse/simpletransformers).


## 1. Training
We use sentence-pair classification model based on _bert-base-chinese_ as our baseline. You can train the baseline model by running 
```
python train.py
```

On each epoch end, the checkpoint will be saved to `outputs/`. The model achieves the best performance on the dev set (splited from train) will be saved to `outputs/best_model/`.

## 2. Evaluation
To evaluate the trained model and generate the submission file, you can simply run
```
python main.py
```
The program will generate the model's evaluation results (_eval_results.txt_) and the submission TXT file (_submission.csv_), which can be found in outputs/. You can also run _eval.py_ to print the evaluation accuracy. 