import torch
import pennylane as qml
from pennylane import numpy as np
import torch.nn as nn
import random
import os
import copy
from random import shuffle
from scipy.optimize import minimize
import shutil
import sys
import pathlib
import time
import math
import configargparse
import matplotlib.pyplot as plt
from typing import Callable, Sequence, Tuple, Union

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', default="PATH_TO_CONFIG_FILE.txt", is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='basedir/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--train_file", type=str, default="train_data.npy", help='training data as .npy file')
    parser.add_argument("--test_file", type=str, default="train_data.npy", help='test data as .npy file')

    # training options
    parser.add_argument("--num_vertices", type=int, default=16, help='no of vertices in each point cloud')
    parser.add_argument("--num_qubits", type=int, default=6, help='no of qubits in the QAE')
    parser.add_argument("--bottleneck_qubits", type=int, default=2, help="no of qubits extra for the bottleneck")
    parser.add_argument("--num_reps", type=int, default=1, help='no of params=num_qubits*(num_reps+1)')
    parser.add_argument("--num_epochs", type=int, default=100, help='no of epochs to run')
    parser.add_argument("--lrate", type=float, default=1e-2, help='initial learning rate')
    parser.add_argument("--basic_block", type=str, default="B", help="Circuit design to be used in the encoder/decoder")
    parser.add_argument("--params_initialization", type=str, default="random", help="Type of parameter initialization to be used")
    parser.add_argument("--patience_updates", type=int, default=100)
    parser.add_argument("--classical_encoder", type=bool, default=False, help="To define the encoder as a classical fully-connected layer")
    parser.add_argument("--classical_decoder", type=bool, default=False, help="To define the decoder as a classical fully-connected layer")
    return parser

def normalize_sv(feature_set):
    norm = qml.math.sum(qml.math.abs(feature_set) ** 2)
    feature_set = feature_set / qml.math.sqrt(norm)
    return feature_set

def data_normalize(data_, num_vertices, min_val, diff_minmax):
    data_ = (data_-min_val)/(diff_minmax)
    aux_vertices = []
    for j in range(num_vertices):
        aux_vertices.append(np.sqrt(3-data_[j,0]**2-data_[j,1]**2-data_[j,2]**2))
    return data_, np.array(aux_vertices)

def define_data(initial_data, num_samples, num_vertices, num_qubits, min_val, diff_minmax):
    data = []
    num_auxzeros = pow(2, num_qubits) - num_vertices*4
    for i in range(num_samples):
        arr = initial_data[i]
        arr, aux_vertices = data_normalize(arr, num_vertices, min_val, diff_minmax)  
        arr = np.append(arr.flatten(), aux_vertices, axis=0)
        arr = np.append(arr, np.zeros((num_auxzeros)), axis=0)
        data.append(normalize_sv(np.asarray(arr)))
    return np.array(data)

def initialize_params(num_qubits, num_reps, initialization_type, basic_block):
    num_parameters = {"A":2*num_qubits, "B":2*num_qubits+8, "C":3*num_qubits, "D":3*num_qubits}
    if initialization_type == "random":
        params = torch.tensor(2*np.pi*np.random.rand((num_parameters[basic_block])*(1)*((2*num_reps))).astype(np.float64), requires_grad=True)
        params = params.reshape(2*num_reps, num_parameters[basic_block])
    elif initialization_type == "identity":
        params=torch.tensor(2*np.pi*np.random.rand((num_parameters[basic_block])*(1)*((num_reps))).astype(np.float64), requires_grad=True)
        params = torch.tensor(torch.cat((params, params), 0), requires_grad=True).reshape(2*num_reps, num_parameters[basic_block])
    return params

def shape_params(num_qubits, num_reps, basic_block):
    num_parameters = {"A":2*num_qubits, "B":2*num_qubits+8, "C":3*num_qubits, "D":3*num_qubits}
    return qml.BasicEntanglerLayers.shape(n_layers=2*num_reps, n_wires=num_parameters[basic_block])

dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev, interface='torch', diff_method='backprop')
def qnode(inputs, weights):
    num_wires_embedding = int(np.log2(inputs.shape[0]))
    qml.QubitStateVector(inputs, wires=range(num_wires_embedding))
    
    wires = list(range(num_qubits))
    for i in range(num_reps):
        if basic_block in ["A", "B"]:
            for j in range(len(wires)):
                qml.RY(weights[i][2*j], wires=[j:j+1])
        if basic_block == "C":
            for j in range(len(wires)):
                qml.RY(weights[i][3*j], wires=[j:j+1])
            for j in range(len(wires)):
                qml.RX(weights[i][3*j+1], wires=[j:j+1])
        if basic_block == "D":
            for j in range(len(wires)):
                qml.RX(weights[i][3*j], wires=[j:j+1])
            for j in range(len(wires)):
                qml.RZ(weights[i][3*j+1], wires= [j:j+1])
        pairs = [(a, b) for a, b in zip(wires, wires[1:]+wires[:1])]
        if basic_block in ["A", "B"]:
            for j in range(len(wires)):
                qml.CRX(weights[i][2*j+1], wires=pairs[j])
        if basic_block in ["C", "D"]:
            for j in range(len(wires)):
                qml.CRX(weights[i][3*j+2], wires=pairs[j])
        if basic_block == "B":
            for k in range(4):
                qml.CRX(weights[i][2*num_qubits+2*k], wires = [k, 4])
                qml.CRX(weights[i][2*num_qubits+2*k+1], wires = [k, 5])
                
        if basic_block in ["A", "B"]:
            for j in range(len(wires)):
                qml.RY(weights[i+num_reps][2*j], wires=[j:j+1])
        if basic_block == "C":
            for j in range(len(wires)):
                qml.RY(weights[i+num_reps][3*j], wires=[j:j+1])
            for j in range(len(wires)):
                qml.RX(weights[i+num_reps][3*j+1], wires=[j:j+1])
        if basic_block == "D":
            for j in range(len(wires)):
                qml.RX(weights[i+num_reps][3*j], wires=[j:j+1])
            for j in range(len(wires)):
                qml.RZ(weights[i+num_reps][3*j+1], wires= [j:j+1])
        pairs = [(a, b) for a, b in zip(wires, wires[1:]+wires[:1])]
        if basic_block in ["A", "B"]:
            for j in range(len(wires)):
                qml.CRX(weights[i+num_reps][2*j+1], wires=pairs[j])
        if basic_block in ["C", "D"]:
            for j in range(len(wires)):
                qml.CRX(weights[i+num_reps][3*j+2], wires=pairs[j])
        if basic_block == "B":
            for k in range(4):
                qml.CRX(weights[i+num_reps][2*num_qubits+2*k], wires = [k, 4])
                qml.CRX(weights[i+num_reps][2*num_qubits+2*k+1], wires = [k, 5])
    return qml.state()

def marginalize_sv(encoder_sv, num_qubits, bottleneck_qubits):
    encoder_sv = encoder_sv**2
    encoder_sv = encoder_sv.reshape(pow(2, num_qubits-bottleneck_qubits), pow(2, bottleneck_qubits)).sum(axis=1).sqrt()
    return encoder_sv
        
class QuantumLayer(torch.nn.Module):
    def __init__(self, qnode, weight_shapes, num_qubits, initialization_type, basic_block):
        super().__init__()
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer_1.weights.data = initialize_params(num_qubits, num_reps, initialization_type, basic_block).float()
        
    def forward(self, x):
        x = torch.abs(self.qlayer_1.forward(x))
        return x

class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if classical_encoder:
            self.custom_1 = nn.Linear(64, 64, bias=False)
        else:
            self.custom_1 = QuantumLayer(qnode, weight_shapes, num_qubits, initialization_type, basic_block)
        if classical_decoder:
            self.custom_2 = nn.Linear(16, 64, bias=False)
        else:
            self.custom_2 = QuantumLayer(qnode, weight_shapes, num_qubits, initialization_type, basic_block)

    def forward(self, x):
        x = x.type(torch.complex128)
        x = torch.abs(self.custom_1(x))
        x = marginalize_sv(x, num_qubits, bottleneck_qubits)
        x = x.type(torch.complex128)
        x = torch.abs(self.custom_2(x)) 
        return x

def cost_iter(data, num_vertices):
    result = model(data)

    result_vertices = result[:3*num_vertices].reshape(num_vertices, 3)
    data_vertices = data[:3*num_vertices].reshape(num_vertices, 3)

    result_aux = result[3*num_vertices:4*num_vertices]
    data_aux = data[3*num_vertices:4*num_vertices]
    norm_val = torch.linalg.norm(result_vertices-data_vertices, axis=-1).mean()
    return norm_val+torch.abs(result_aux-data_aux).mean()

def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def _write_mesh(vertices, mesh_file):
	mesh_string = ""
	for x, y, z in vertices:
		mesh_string += "v " + str(x) + " " + str( y ) + " " + str(z) + "\n"
	with open(mesh_file, "w") as mesh_file:
		mesh_file.write(mesh_string)
  
def generate_mesh_examples(test_data, num_vertices, diff_minmax, min_val, basedir, exp_name, mesh_type):
    for i in range(0, 2000, 6):
        input = test_data[i].clone()
        result = model(input)

        input = input[:3*num_vertices].reshape(num_vertices, 3)
        input *= np.sqrt(3*num_vertices)
        input *= (diff_minmax)
        input += min_val

        result = result[:3*num_vertices].reshape(num_vertices, 3)
        result = result.detach().numpy()
        result *= np.sqrt(3*num_vertices)
        result *= (diff_minmax)
        result += min_val

        _write_mesh(input, basedir+exp_name+"test_original_mesh_"+str(i)+".obj")
        _write_mesh(result, basedir+exp_name+"test_"+mesh_type+"_predicted_"+str(i)+".obj")
        
def calculate_euclidean_distances(data, num_vertices, diff_minmax, min_val, data_type, basedir, exp_name):
    euclidean_distances = []
    predicted_meshes = []

    for i in range(len(data)):
        input = data.clone()[i]
        result = model(data[i])

        input = input[:3*num_vertices].reshape(num_vertices, 3)
        input *= np.sqrt(3*num_vertices)
        input *= (diff_minmax)
        input += min_val
    
        result = result[:3*num_vertices].reshape(num_vertices, 3)
        result = result.detach().numpy()
        result *= np.sqrt(3*num_vertices)
        result *= (diff_minmax)
        result += min_val

        predicted_meshes.extend(result)
        euclidean_distances.append(np.linalg.norm(np.array(input)-np.array(result), axis=-1).mean())

    f4 = basedir+data_type+'_predicted_meshes_'+exp_name+'.npy'
    np.save(f4, np.array(predicted_meshes))
    f5 = basedir+data_type+'_euclidean_distances_'+exp_name+'.npy'
    np.save(f5, np.array(euclidean_distances))
    
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    create_folder(args.basedir)
    basedir = args.basedir
    
    exp_name = args.expname
    num_vertices = args.num_vertices
    patience_updates = args.patience_updates
    
    global num_qubits, bottleneck_qubits, num_reps, basic_block, initialization_type, weight_shapes, classical_encoder, classical_decoder
    num_qubits = args.num_qubits  
    bottleneck_qubits = args.bottleneck_qubits
    num_reps = args.num_reps
    basic_block = args.basic_block
    initialization_type = args.params_initialization
    weight_shapes = {"weights":shape_params(num_qubits, num_reps, basic_block)}
    classical_encoder = args.classical_encoder
    classical_decoder = args.classical_decoder

    model = HybridModel()
    
    initial_train = np.load(args.train_file)
    initial_test = np.load(args.test_file)
    combined_data = np.concatenate((initial_train, initial_test), axis=0)
    min_val = np.amin(combined_data[:,:,:], axis=(0, 1)) 
    max_val = np.amax(combined_data[:,:,:], axis=(0, 1))
    diff_minmax = np.max(max_val-min_val)
    
    train_data = define_data(initial_train, len(initial_train), num_vertices, num_qubits, min_val, diff_minmax)
    train_data = torch.tensor(train_data, requires_grad=False)    
    test_data = define_data(initial_test, len(initial_test), num_vertices, num_qubits, min_val, diff_minmax)
    test_data = torch.tensor(test_data, requires_grad=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lrate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience_updates, verbose=True, min_lr=1e-4)

    if os.path.exists(basedir+"latest.pth"):
        checkpoint = torch.load(basedir+"latest.pth")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initial_epoch = checkpoint["epoch"]
        initial_iter = checkpoint["iteration"]
        loss_list = checkpoint["loss_list"]
        scheduler_loss = checkpoint["scheduler_loss"]
        loss_epoch = checkpoint["loss_epoch"]
    else:
        torch.save(model.state_dict(), basedir+"model_dict_init.pth")
        initial_epoch = 0
        initial_iter = 0
        loss_list = []
        loss_epoch = 0
        scheduler_loss = 0

    if not os.path.exists(basedir+exp_name+"test_original_mesh_0.obj"):
        generate_mesh_examples(test_data, num_vertices, diff_minmax, min_val, basedir, exp_name, "initial")

    num_epochs = args.num_epochs

    for i in range(initial_epoch, num_epochs):
        
        if i > initial_epoch:
            initial_iter = 0
            loss_epoch = 0
            scheduler_loss = 0
        
        for j in range(initial_iter, len(train_data)):
            opt.zero_grad()
            loss = cost_iter(train_data[j], num_vertices)
            loss.backward()
            opt.step()
            
            loss_epoch += loss.detach()
            scheduler_loss += loss.detach()
        
            if j % 50 == 0:
                scheduler.step(scheduler_loss/50)
                scheduler_loss = 0

            if (j+i*len(train_data)) % 500==0:
                checkpoint = {}
                checkpoint["model"] = model.state_dict()
                checkpoint["opt"] = opt.state_dict()
                checkpoint["scheduler"] = scheduler.state_dict()
                checkpoint["epoch"] = i
                checkpoint["iteration"] = j
                checkpoint["loss_list"] = loss_list
                checkpoint["scheduler_loss"] = scheduler_loss
                checkpoint["loss_epoch"] = loss_epoch
                torch.save(checkpoint, basedir+"latest.pth")
                
        loss_list.append(loss_epoch/len(train_data))
        if i%20 == 0:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["opt"] = opt.state_dict()
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = i
            checkpoint["iteration"] = j
            checkpoint["loss_list"] = loss_list
            checkpoint["scheduler_loss"] = scheduler_loss
            checkpoint["loss_epoch"] = loss_epoch
            torch.save(checkpoint, basedir+"latest_"+str(i)+".pth")
    
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["opt"] = opt.state_dict()
    checkpoint["scheduler"] = scheduler.state_dict()
    checkpoint["epoch"] = i
    checkpoint["iteration"] = j
    checkpoint["loss_list"] = loss_list
    checkpoint["scheduler_loss"] = scheduler_loss
    checkpoint["loss_epoch"] = loss_epoch
    torch.save(checkpoint, basedir+"latest.pth")

    print('###### Loss List ######')
    plt.plot(np.arange(1, len(loss_list)+1, 1), loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(exp_name)
    plt.savefig(basedir+exp_name+".png")
    
    f1 = basedir+'loss_'+exp_name+'.npy'
    np.save(f1, loss_list)
    
    generate_mesh_examples(test_data, num_vertices, diff_minmax, min_val, basedir, exp_name, "final")

    calculate_euclidean_distances(train_data, num_vertices, diff_minmax, min_val, "train", basedir, exp_name)
    calculate_euclidean_distances(test_data, num_vertices, diff_minmax, min_val, "test", basedir, exp_name)
