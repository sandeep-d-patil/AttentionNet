# LaneDetection

This repository is an example of LaneDetection method which uses attention based LSTM model. To train the model, use


1. Clone the github repository
2. Download the dataset (TuSimple or LLAMAS dataset) to dataset folder
3. Create the train_indices and valid_indices and place them on the root folder of the repository.

### Download datasets

Download training and testset

```
cd ~/LaneDetection/data
gdown --id "1LLJwNidd7SGT4tSDnIaEEnlzCMMTqqNY&confirm=t"
gdown --id "1k04eD3Ieoq-NcSN2j40uFowpHN6nxa1-&confirm=t"
unzip trainset.zip
unzip testset.zip
```
To start the training
```
cd LaneDetection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
python3 train.py --model 'UNet_Attention' --epochs 100 --lr 0.01 --batch-size 16
```

Evaluate tuSimple dataset
```
cd LaneDetection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
python3 test.py --evaluate
```

To cite this work please use the citation below

    @article{patil-2022,
        author = {Patil, Sandeep and Dong, Yongqi and Farah, Haneen and Hellendoorn, Hans},
        journal = {Social Science Research Network},
        month = {1},
        title = {{Sequential Neural Network Model with Spatial-Temporal Attention Mechanism for Robust Lane Detection Using Multi Continuous Image Frames}},
        year = {2022},
        doi = {10.2139/ssrn.4273506},
        url = {https://doi.org/10.2139/ssrn.4273506},
    }