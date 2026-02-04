import json
import copy
import os
import argparse
import random
from tqdm import trange
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.nn as nn
from torch import optim
from copy import deepcopy
from clients import ClientsGroup, client
from Models import CNN
from hypernet import HyperNetwork

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('--dataset', type=str, default='cifar', help='options:[mnist, cifar, cifar100, miniImagenet]')
parser.add_argument('--num_of_clients', type=int, default=30, help='numer of the clients')
parser.add_argument('--num_in_comm', type=int, default=10,
                    help='number of communication clients in each round (default: 5)')
parser.add_argument('--m', type=int, default=10, metavar='Send',
                    help='number of models need to be sent to each client(default: 5)')
parser.add_argument('--way', type=int, default=5 , metavar='TESTWAY',
                    help='number of classes per client(default: 5)')
parser.add_argument('--shot', type=int, default=5, metavar='SHOT',
                    help='number of support samples per class')
parser.add_argument('-iid', '--IID', type=int, default=1,
                    help='the way to allocate data to clients (1:pathological, 2:realistic)')

parser.add_argument('--alpha', type=float, default=1, metavar='ALPHA',
                    help='dirichlet parameters (default: 1)')
parser.add_argument('--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('--model_name', type=str, default='cnn', help='the model to train')
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="learning rate, use value from origin paper as default")
parser.add_argument('--num_comm', type=int, default=500, help='number of communications')

parser.add_argument('--lambda_', type=float, default=0.0005,
                    help='l1-norm regularization parameters (default: 0.00001) ')
parser.add_argument('--round', type=int, default=25, metavar='Round',
                    help='rounds started to update Q (default: 25)')

def main():
    args = parser.parse_args()
    args = args.__dict__
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

    net = None
    if args['model_name'] == 'cnn':
        if args['dataset'] == 'mnist':
            net = CNN(x_dim=1)
        elif args['dataset'] == 'cifar':
            net = CNN(x_dim=3)
        elif args['dataset'] == 'cifar100':
            net = CNN(x_dim=3)
        elif args['dataset'] == 'miniImagenet':
            net = CNN(x_dim=3)

    net = net.to(dev)
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev, args['way'], args['shot'],
                              args['alpha'], net, args['num_in_comm'], args['num_of_clients'])

    performance_Q = np.zeros((args['num_of_clients'], args['num_of_clients']))
    

    global_parameters_start = {}
    global_parameters_end = {}
    global_parameters = {}

    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_of_clients']):
        global_parameters_start['client{}'.format(i)] = copy.deepcopy(global_parameters)
        myClients.clients_set['client{}'.format(i)].set_parameter(global_parameters)
    num_in_comm = args['num_in_comm']

    with torch.no_grad():
        losses = []
        accs = []
        for client in range(args['num_of_clients']):
            loss, acc = myClients.clients_set['client{}'.format(client)].local_test_2(net)
            losses.append(loss)
            accs.append(acc)
        print('Before Train: loss={:.3f}±{:.3f}, acc={:.3f}±{:.3f}'.format(np.mean(losses), np.std(losses, ddof=1),
                                                                     np.mean(accs), np.std(accs, ddof=1)))

    acc_all = [np.mean(accs)]
    for i in trange(args['num_comm']):
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = [i for i in order[0:num_in_comm]]

        for client in clients_in_comm:
            sends_index = []
            models_to_send = []
            Q = performance_Q[client].copy()
            Q[client] = -np.inf
            selected_index = list(range(args['num_of_clients']))
            selected_index.remove(client)
            for j in range(num_in_comm):
                max_value = np.max(Q)
                indexs = np.where(Q == max_value)[0].tolist()
                index = random.sample(indexs, 1)[0]
                selected_index.remove(index)
                sends_index.append(index)
                models_to_send.append(global_parameters_start['client{}'.format(index)])
                Q[index] = -np.inf

            Net, weight, loss, acc, diff_para= myClients.clients_set['client{}'.format(client)].\
                localUpdate(args['epoch'], net, models_to_send, opti, args['lambda_'],client_id = client)

            parameters = {}
            for key, var in Net.state_dict().items():
                parameters[key] = var.clone()
            global_parameters_end['client{}'.format(client)] = parameters
            assert weight.shape[0] == len(sends_index)
            if i >= args['round']:
                for j in range(args['m']):
                    performance_Q[client][sends_index[j]] += weight[j]
        for client in clients_in_comm:
            global_parameters_start['client{}'.format(client)] = \
                copy.deepcopy(global_parameters_end['client{}'.format(client)])
        losses = []
        accs = []
        if i % 10 == 0:
            for client in range(args['num_of_clients']):
                loss, acc = myClients.clients_set['client{}'.format(client)].local_test(Net)
                losses.append(loss)
                accs.append(acc)
            acc_all.append(np.mean(accs))

    losses = []
    accs = []
    for client in range(args['num_of_clients']):
        loss, acc = myClients.clients_set['client{}'.format(client)].local_test(Net)
        losses.append(loss)
        accs.append(acc)
    acc_all.append(np.mean(accs))

    print('Test: acc={:.2f}'.format(acc_all[-1] * 100))
    results = {
        'parameters': args,
        'accuracy': acc_all[-1] * 100,
        'accuracy_all': acc_all
    }
    return acc_all[-1], results





def convert_to_serializable(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
def save_results(final_results):
    os.makedirs("../result", exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"../result/final_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(final_results, f, indent=4, default=convert_to_serializable)
    
    print(f"Results saved in {filename}")

if __name__ == "__main__":
    accs = []
    final_results = {
        "individual_results": [],
        "final_test": {}
    }

    try:
        for i in range(5):
            acc, result = main()
            accs.append(acc)
            final_results["individual_results"].append(result)
    finally:
        final_mean = np.mean(accs) * 100
        final_std = np.std(accs, ddof=1) * 100
        final_results["final_test"] = {
            "final_accuracy_mean": final_mean,
            "final_accuracy_std": final_std
        }
        print('Final Test: acc={:.2f}±{:.2f}'.format(final_mean, final_std))
        save_results(final_results)