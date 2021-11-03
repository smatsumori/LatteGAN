# LatteGAN
## Prerequisites
### libraries

- cuda10.1, cudnn7, ubuntu.
- pipenv for `Pipfile`.
  - refer it for the detail of python version or libraries.

### datasets
Download GeNeVA datasets, including CoDraw and i-CLEVR, from [the project page](https://www.microsoft.com/en-us/research/project/generative-neural-visual-artist-geneva/).
After the extraction, your directory structure should look like this

```bash
./data/
├── CoDraw/
│   ├── codraw_test.h5
│   ├── codraw_train.h5
│   ├── codraw_val.h5
│   ├── objects.txt
├── iCLEVR/
│   ├── clevr_test.h5
│   ├── clevr_train.h5
│   ├── clevr_val.h5
│   ├── objects.txt
├── misc
│   └── glove_codraw_iclevr.txt
└── models/
    ├── codraw_inception_best_checkpoint.pth
    └── iclevr_inception_best_checkpoint.pth
```

## Pretraining
Execute the following scripts to finetune the text feature extractor BERT-base-uncased.
The artifacts of these scripts will be stored under `./results/experiments/exp(78|95)-pretrain-tirg/`.

```bash
# for CoDraw
pipenv run python src/main.py --yaml_path ./params/exp078.yaml --pretrain_tirg --gpu_ids=0

# for i-CLEVR
pipenv run python src/main.py --yaml_path ./params/exp095.yaml --pretrain_tirg --gpu_ids=0
```

Execute the following scripts to prepare the embeddings of instructions.
Note that the arguments `model_path` in the yaml files should be set the path of the weights acquired the above pretraining scripts.

```bash
# for CoDraw
pipenv run python src/main.py --yaml_path ./params/exp085.yaml --create_embs_from_model --gpu_ids=0

# for i-CLEVR
pipenv run python src/main.py --yaml_path ./params/exp098.yaml --create_embs_from_model --gpu_ids=0
```

## Adversarial Training
Execute following scripts to train LatteGAN.
Generation of images and calculation of metrics AP, AR, F1, and RSIM will be conducted simultaneously during training.
All of the artifacts will be stored under `./results/experiments/` with the corresponding experiment numbers.

```bash
# CoDraw: LatteGAN
pipenv run python src/main.py --yaml_path ./params/exp169.yaml --train_propv1_scain_geneva --gpu_ids=0,1,2,3

# iCLEVR: LatteGAN
pipenv run python src/main.py --yaml_path ./params/exp170.yaml --train_propv1_scain_geneva --gpu_ids=0,1,2,3
```

The results of the paper can be reproduced by executing the above scripts with the following hardware settings.

- 4x NVIDIA Tesla V100 (SXM2) GPUs
- 2x Intel Xeon Gold 6148 (27.5M cache, 2.40 GHz, 20 cores) processors
- 384GiB of memory
