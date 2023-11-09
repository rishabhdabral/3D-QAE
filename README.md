# 3D-QAE

This is the official repository for the project "3D-QAE: A Fully Quantum Auto-Encoder for 3D Shapes". This project introduces the first quantum auto-encoder for 3D point sets, using data normalisation techniques and by introducing auxiliary values to combat the input and output restrictions of quantum circuits. For more details, refer to the [project page](https://4dqv.mpi-inf.mpg.de/QAE3D/).

The codebase is implemented using the [Pytorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html) in Pennylane, for noise-less quantum simulation on the CPU.

<p align="center">
<img src='https://github.com/rishabhdabral/3D-QAE/blob/main/Images/teaser_figure.png' alt='Teasure Figure' width=400pt/>
</p>

## Create environment

The code can be run by creating an environment ```torch``` using the given environment.yml file (or installing the requirements using ```pip```). 

We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up the environment:
```
conda env create -f environment.yml
```
This conda environment can then be activate with ```conda activate torch```

## Dataset
#### Downloading the data 
The data can be downloaded from the [AMASS website](https://amass.is.tue.mpg.de/download.php) and saved in a folder named as ```Amass_files```. For this project we used the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/) downloaded as .npz files. 

#### Downloading the SMPLify-X model
To extract the joints from the motion capture data, we need [SMPLify-X](https://smpl-x.is.tue.mpg.de/index.html) models which can be downloaded from their website. The path for this downloaded model needs to be updated inside the ```find_joints.py``` file in the ```bm_fname``` variable.

#### Processing AMASS data
Next, we [extract joints](https://github.com/nghorbani/human_body_prior) from the AMASS data files . To do this activate the ```torch``` environment and run,
```
python3 find_joints.py
```
## Training and Testing
After the dataset preparation, the quantum models can be trained by running the given python files. 
- To train with a repeat architecture, run: ```python repeat_architecture.py  --config <path_to_config_file>```
- To train with an identity architecture, run: ```python identity_architecture.py  --config <path_to_config_file>```

The config file allows us to set the training options. 
- The parameter initialisation for the model can be set as random or identity in the ```params_initialization``` argument. For example, to set the parameter initialisation as random, run: ```python <filename.py> --config <path_to_config_file> --params_initialization random```
- The basic building blocks of the quantum circuit can be chosen out of "A", "B", "C" or "D" in the ```basic_block``` argument. For example, the ```basic_block``` "B" can be chosen, by running: ```python <filename.py> --config <path_to_config_file> --basic_block B```. <img src='https://github.com/rishabhdabral/3D-QAE/blob/main/Images/all_circuits.png' alt='Basic blocks we investigate'/> 
- The number of training epochs can be set using ```num_epochs``` argument.
- The number of blocks each in the encoder/decoder of the model is set using ```num_reps``` argument.
- The encoder or decoder can be replaced by classical fully-connected layers to build a hybrid model by setting the corresponding arguments. For example to build a hybrid model with classical encoder and quantum decoder, run: ```python <filename.py> --config <path_to_config_file> --classical_encoder True```

## Best Results
Evaluating the different combinations, we observe that the basic block "B" alongwith the "repeat" architecture and "identity" parameter initialisation scheme works the best. Using 8 blocks each in the encoder and decoder, we report a mean euclidean distance of 10.86 cm. 

## Citation
```
@InProceedings{Rathi2023, 
    author={Rathi, Lakshika  and Tretschk, Edith and Theobalt, Christian and Dabral, Rishabh  and Golyanik, Vladislav}, 
    title={{3D-QAE}: Fully Quantum Auto-Encoding of 3D Point Clouds}, 
    booktitle={The British Machine Vision Conference (BMVC)}, 
    year={2023} 
}
