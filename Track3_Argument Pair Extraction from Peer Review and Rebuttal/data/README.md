# RR dataset

## RR-submission-v2

### Data Processing
To process the data, we adopt [bert-as-service](https://github.com/hanxiao/bert-as-service) as a tool to obtain the embeddings for all tokens [x0, x1, · · · , xT −1] in the sentence.

#### Install
```
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
```

#### Download a pre-trained BERT model
e.g. Download a [model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip), then uncompress the zip file into some folder, say ```/tmp/english_L-12_H-768_A-12/```

#### Start the BERT service
```bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -max_seq_len NONE -pooling_strategy NONE```

#### Use Client to Get Sentence Encodes
Run ```./dataProcessing.py```.

Now you will get ```vec_train.pkl```, ```vec_dev.pkl```, ```vec_test.pkl```.

## Citation
```
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```
