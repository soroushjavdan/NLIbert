# NLIbert 

Dialogue Natural Language Inference with bert classifier.

### Description
----------

In this code we make use of dataset called [Dialogue NLI](https://wellecks.github.io/dialogue_nli/). 

How to:
-------

#### First
you need to download the DataSet from the below link.
```link 
https://wellecks.github.io/dialogue_nli/
``` 
Then place the downloaded data inside ```data/dialogue_nli_extra``` directory. 

#### Second
because of problem during compression process the training file need to be repair. run the below command in order to repair it.
```console
❱❱❱ python data_repairer.py
```
#### Third
install the requirements
```console
❱❱❱ pip install requirements.txt
```
Now we are good to go !!

#### Training phase
just run the below command
```console
❱❱❱ python train.py --gpu --data_path data/dialogue_nli_extra/ --save_path save/ --lr 5e-5 --batch_size 32 --epochs 4 --plot_path save/plot/ --bert_model bert-base-cased
```

