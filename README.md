# AttentionNet: Robust Lane Detection

## Introduction

AttentionNet is an innovative Lane Detection method utilizing an attention-based LSTM model based on the publication https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4282960 . This project aims to provide a robust solution for lane detection by leveraging the power of sequential neural networks and spatial-temporal attention mechanisms. The model is designed to work with multi-continuous image frames, making it suitable for real-time applications in autonomous driving and traffic monitoring systems.

## LaneDetection

This repository is an example of LaneDetection method which uses attention based LSTM model. To train the model, use


## Installation

To set up the AttentionNet project for development or testing, follow these steps:

1. Clone the github repository
2. Download the dataset (TuSimple or LLAMAS dataset) to dataset folder
3. Start the training

1. **Clone the repository:**
```bash
git clone https://github.com/sandeep-d-patil/AttentionNet.git
```

2. **Download datasets:**

Download training and testset

```bash
cd ~/LaneDetection/data
gdown --id "1LLJwNidd7SGT4tSDnIaEEnlzCMMTqqNY&confirm=t"
gdown --id "1k04eD3Ieoq-NcSN2j40uFowpHN6nxa1-&confirm=t"
unzip trainset.zip
unzip testset.zip
```

3. **Start the training:**

```bash
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
Contributing
Contributions to AttentionNet are welcome! If you have suggestions for improvements or bug fixes, please feel free to:
Fork the repository.
Create a new branch for your feature or fix.
Commit your changes.
Push to the branch.
Submit a pull request.
Please ensure your code adheres to the project's coding standards and include tests for new features or fixes.
Citation
If you use AttentionNet in your research, please cite the following work:

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

### License

This project is licensed under the terms of the MIT license.

### Contact

For any questions or concerns regarding AttentionNet, please open an issue in the repository or contact the maintainers directly through GitHub.


This README template provides a clear and structured overview of the AttentionNet project, making it easier for users and contributors to understand the project's purpose, set up their development environment, use the project, and contribute to its development.
