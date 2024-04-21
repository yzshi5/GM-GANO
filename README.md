# cGM-GANO
![demo](https://github.com/yzshi5/GM-GANO/blob/main/model.png)
The repository contains code for [Broadband Ground Motion Synthesis via Generative Adversarial Neural Operators: Development and Validation](https://pubs.geoscienceworld.org/ssa/bssa/article/doi/10.1785/0120230207/636448/Broadband-Ground-Motion-Synthesis-via-Generative) (also available on [arxiv](https://arxiv.org/abs/2309.03447))
, for more information about GANO implementation, please refer to [GANO](https://github.com/neuraloperator/GANO)

## Installation

create conda environment and install necessary libraries
```
conda create --name gano

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c anaconda ipykernel

pip install ipykernel

python -m ipykernel install --user --name=gano

conda install pandas

conda install matplotlib
 
pip install scipy

pip install tqdm

pip install opencv-python
```

## Load the pretrained model
please download the trained model through following link,  and store the model under `kik_net_trained_model` folder
https://drive.google.com/file/d/18k366Y4UmaGoYxepwzZaGo_nw6Kup0cW/view?usp=sharing

## Quick start of using trained GANO
run `tutorials_for_cGmGANO.ipynb` file for generating 3C waveforms 

## Train your own cGM-GANO
run `train_GANO.ipynb`
