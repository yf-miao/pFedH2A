import os
import copy
import time
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from getData import GetDataSet
from random import randint
from hypernet import HyperNetwork
import argparse
from utils import euclidean_metric
from copy import deepcopy


class client(object):
    def __init__(self, train_data, train_label, test_data, test_label, dev, net, num_in_comm, num_of_clients):
        parser = argparse.ArgumentParser()

        parser.add_argument("--k", type=int, default=0)
        parser.add_argument("--hn_lr", type=float, default=5e-3)
        parser.add_argument("--hn_momentum", type=float, default=0.0)
        parser.add_argument("--embedding_dim", type=int, default=100)
        parser.add_argument("--hidden_dim", type=int, default=100)
        
        args = parser.parse_args()
        
        self.dev = dev
        self.train_label = train_label.to(self.dev)
        self.test_label = test_label.to(self.dev)
        self.train_ds = train_data.to(self.dev)
        self.test_ds = test_data.to(self.dev)
        self.clusters = 5
        self.net = net.to(self.dev)
        self.num_in_comm = num_in_comm
        self.num_of_clients = num_of_clients

        labels = list(set(train_label.numpy()))
        for i, label in enumerate(labels):
            self.train_label[torch.where(train_label == label)] = i
            self.test_label[torch.where(test_label == label)] = i
        self.num_class = len(labels)
        self.num_samples = []

        self.local_parameters = {}
        self.last_loss = None
        self.number_of_sample = torch.zeros((self.train_label.shape[0], self.num_class), dtype=torch.float32,
                                            requires_grad=False).to(self.dev)
        for j in range(self.num_class):
            num_sample = torch.where(self.train_label == j)[0].shape[0]
            self.num_samples.append(num_sample)
            for i in range(self.number_of_sample.shape[0]):
                if self.train_label[i] == j and num_sample - 1 != 0:
                    self.number_of_sample[i][j] = num_sample - 1
                else:
                    self.number_of_sample[i][j] = num_sample
                    
        self.hypernet = HyperNetwork(
            embedding_dim=args.embedding_dim,
            layer_num=len(net.state_dict()),
            client_num=self.num_of_clients,
            selected_num = self.num_in_comm+1,
            hidden_dim=args.hidden_dim,
            K=args.k,
        )
        
        self.hypernet_opti = torch.optim.SGD(
            self.hypernet.parameters(),
            lr=args.hn_lr,
            momentum=args.hn_momentum,
        )
    
        self.clients_hypernet_params = {
            client_id: deepcopy(self.hypernet.state_dict()) for client_id in range(num_of_clients)
        }

    def set_parameter(self, parameters):
        for key, var in parameters.items():
            self.local_parameters[key] = var.clone()
    
    def updateHypernet(self,client_id, diff_para, aggregation_parameters):
        self.hypernet_opti.zero_grad()
        hn_grads = torch.autograd.grad(
            outputs = list(aggregation_parameters.values()),
            inputs = self.hypernet.parameters(),
            grad_outputs=[
                -diff
                for diff in diff_para.values()
            ],
            allow_unused=True,
        )
        for param, grad in zip(self.hypernet.parameters(), hn_grads):
            if grad is not None:
                param.grad = grad
        self.hypernet_opti.step()
        self.clients_hypernet_params[client_id] = deepcopy(
            self.hypernet.state_dict()
        )        

    def weight_l2(self, weight):
        weight_l2 = {}
        for client in range(len(weight[0])):
            total_norm = 0
            for layer in range(len(weight)):
                total_norm += weight[layer][client].norm(2).item()
            weight_l2[client] = total_norm
        weight_l2 = np.array(list(weight_l2.values()))
        return weight_l2

    def localUpdate(self, localEpoch, Net, models_to_send, optim, lambda_,client_id):
    
        self.hypernet.load_state_dict(self.clients_hypernet_params[client_id])
        
        losses = []

        models_to_send.append(self.local_parameters)
        with torch.no_grad():
            for i in range(len(models_to_send)):
                Net.load_state_dict(models_to_send[i], strict=True)
                _, _, acc, loss = self.calculate_loss(Net, lambda_)
                losses.append(-loss.item())
        losses = torch.tensor(losses)

        weight = self.hypernet(client_id, losses)
        for i in range(len(Net.state_dict())):
            weight[i] = F.softmax(weight[i], dim=0)
        
        aggregation_parameters = None
        
        for layer_idx, key in enumerate(models_to_send[0].keys()):
            if aggregation_parameters is None:
                aggregation_parameters = {}

            for client_idx, parameters in enumerate(models_to_send):
                var = parameters[key]
                current_weight = weight[layer_idx][client_idx]

                if key not in aggregation_parameters:
                    aggregation_parameters[key] = var.clone() * current_weight
                else:
                    aggregation_parameters[key] += var.clone() * current_weight


        Net.load_state_dict(aggregation_parameters, strict=True)
        Net.train()

        for epoch in range(localEpoch):
            optim.zero_grad()
            loss1, loss2, acc_val, _ = self.calculate_loss(Net, lambda_)
            loss1.backward()
            optim.step()
        
        model_params_diff = {}
        for key, var in Net.state_dict().items():
            model_params_diff[key] = self.local_parameters[key] - aggregation_parameters[key]
            self.local_parameters[key] = var.clone()
        
        self.updateHypernet(client_id,model_params_diff, aggregation_parameters)

        weight = self.weight_l2(weight)
        weights = weight[:len(models_to_send)-1] - weight[len(models_to_send)-1]

        return Net, weights.copy(), loss1.item(), acc_val, model_params_diff

    def calculate_loss(self, Net, lambda_):

        x_embedding = Net(self.train_ds)
        dis_ = euclidean_metric(x_embedding, x_embedding)
        distance = torch.zeros((x_embedding.shape[0], self.num_class), dtype=torch.float32, requires_grad=True).to(
            self.dev)

        def cal_cal(j):
            a = torch.where(self.train_label == j)[0].unsqueeze(0)
            a = a.expand(x_embedding.shape[0], a.shape[1])
            b = dis_.gather(1, a)
            c = b.sum(axis=1)
            distance[:, j] = c

        for j in range(self.num_class):
            cal_cal(j)

        avg_dis = torch.div(distance, self.number_of_sample)

        log_p = F.log_softmax(avg_dis, dim=1)

        loss = - log_p.gather(1, self.train_label.unsqueeze(dim=1)).mean()
        _, y_hat = log_p.max(1)
        acc = torch.eq(y_hat, self.train_label).float().mean()

        l1_regularization = torch.tensor([0], dtype=torch.float32).to(self.dev)
        l2_regularization = torch.tensor(0.0, dtype=torch.float32, device=self.dev)
        for param in Net.parameters():
            l1_regularization += torch.norm(param, 1)
            l2_regularization += torch.norm(param, 2)

        loss_l1 = loss + lambda_*l1_regularization
        loss_l2 = loss + lambda_*l2_regularization
        return loss_l1, loss_l2, acc.cpu(), loss

    def local_test(self, Net):
        Net.load_state_dict(self.local_parameters, strict=True)
        Net.eval()
        train_data_embedding = Net(self.train_ds)
        test_data_embedding = Net(self.test_ds)
        dis_ = euclidean_metric(test_data_embedding, train_data_embedding)
        distance = torch.zeros((test_data_embedding.shape[0], self.num_class), dtype=torch.float32, requires_grad=False) \
            .to(self.dev)

        def cal_cal(j):
            a = torch.where(self.train_label == j)[0].unsqueeze(0)
            a = a.expand(test_data_embedding.shape[0], a.shape[1])
            b = dis_.gather(1, a)
            c = b.mean(axis=1)
            distance[:, j] = c

        for j in range(self.num_class):
            cal_cal(j)
        p = F.softmax(distance, dim=1)
        log_p = F.log_softmax(distance, dim=1)

        c = self.test_label.unsqueeze(dim=1)
        loss = - log_p.gather(1, self.test_label.unsqueeze(dim=1)).mean()
        _, y_hat = log_p.max(1)
        acc_test = torch.eq(y_hat, self.test_label).float().mean()
        return loss.item(), acc_test.cpu()

    def local_test_2(self, Net): 
        Net.eval()
        train_data_embedding = Net(self.train_ds)
        test_data_embedding = Net(self.test_ds)
        dis_ = euclidean_metric(test_data_embedding, train_data_embedding)
        distance = torch.zeros((test_data_embedding.shape[0], self.num_class), dtype=torch.float32, requires_grad=False)\
            .to(self.dev)

        def cal_cal(j):
            a = torch.where(self.train_label == j)[0].unsqueeze(0)
            a = a.expand(test_data_embedding.shape[0], a.shape[1])
            b = dis_.gather(1, a)
            c = b.mean(axis=1)
            distance[:, j] = c

        for j in range(self.num_class):
            cal_cal(j)
        p = F.softmax(distance, dim=1)
        log_p = F.log_softmax(distance, dim=1)

        c = self.test_label.unsqueeze(dim=1)
        loss = - log_p.gather(1, self.test_label.unsqueeze(dim=1)).mean()
        _, y_hat = log_p.max(1)
        acc_test = torch.eq(y_hat, self.test_label).float().mean()
        return loss.item(), acc_test.cpu()


def split_noniid(train_labels, alpha, n_clients):

    classes = list(set(train_labels))
    n_classes = len(classes)

    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in classes]

    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        _ = np.cumsum(fracs)[:-1]
        temps = np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))
        for i in range(len(temps)):
            temps[i] = temps[i].tolist()
        max_num = 0
        max_index = -1
        for i, temp in enumerate(temps):
            if max_num < len(temp):
                max_index = i
                max_num = len(temp)
        for i, temp in enumerate(temps):
            if len(temp) == 1:
                temps[i].append(temps[max_index].pop())

        for i, temp in enumerate(temps):
            temps[i] = [np.array(temps[i], dtype=int)]

        for i, idcs in enumerate(temps):
            client_idcs[i] += idcs

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, way, shot, alpha, net ,num_in_comm, num_of_clients):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.way = way
        self.shot = shot
        self.clients_set = {}
        self.alpha = alpha
        self.clusters = 5
        self.net = net
        self.num_in_comm = num_in_comm
        self.num_of_clients = num_of_clients

        self.test_data_loader = None
        if isIID == 2:
            self.dataSetDirichletAllocation()
        else:
            self.dataSetBalanceAllocation()

    def dataSetDirichletAllocation(self):
        DataSet = GetDataSet(self.data_set_name, self.is_iid)
        train_data = DataSet.train_data
        train_label = DataSet.train_label
        train_label_value = np.argmax(train_label, axis=1)

        all_index = {}
        train_index = {}
        test_index = {}
        for i in range(train_label.shape[1]):
            all_index[i] = []
            train_index[i] = []
            test_index[i] = []
        for i in range(len(train_label_value)):
            all_index[train_label_value[i]].append(i)
        index = 0
        num_per_class = int(self.shot*self.num_of_clients/self.clusters)
        ns = self.produce_clusters(train_label.shape[1])
        for i in range(self.clusters):
            for j in range(train_label.shape[1]):
                train_index[i] = []
            for j in ns[i]:
                num_samples = min(num_per_class, len(all_index[j]))
                train_index[j] = random.sample(all_index[j], num_samples)
                for k in train_index[j]:
                    all_index[j].remove(k)
            indexs = []
            for j in ns[i]:
                indexs.append(train_index[j])
            indexs = np.concatenate(indexs)
            labels = train_label_value[indexs.astype(int)]
            client_train_ids = split_noniid(labels, self.alpha, int(self.num_of_clients/self.clusters))
            for j, cur in enumerate(client_train_ids):
                client_train_ids[j] = indexs[client_train_ids[j]]
            for cur_id in client_train_ids:
                local_train_label = train_label_value[cur_id.astype(int)]
                local_train_data = train_data[cur_id.astype(int)]
                cur_labels = list(set(local_train_label))
                cur_test_ids = []
                for cur_label in cur_labels:
                    num_query = np.argwhere(local_train_label == cur_label).shape[0]
                    num_sample = min(num_query * 2, len(all_index[cur_label]))
                    _ = random.sample(all_index[cur_label], num_sample)

                    cur_test_ids.append(_)
                    for __ in _:
                        all_index[cur_label].remove(__)
                cur_test_ids = np.concatenate(cur_test_ids)
                local_test_label = train_label_value[cur_test_ids.astype(int)]
                local_test_data = train_data[cur_test_ids.astype(int)]
                someone = client(torch.tensor(local_train_data), torch.tensor(local_train_label),
                                 torch.tensor(local_test_data), torch.tensor(local_test_label), self.dev, self.net, self.num_in_comm, self.num_of_clients)
                self.clients_set['client{}'.format(index)] = someone
                index += 1

    def produce_clusters(self, num_class):
        ns = []
        for i in range(self.clusters):
            index = random.sample(range(0, num_class), self.way)
            ns.append(index)
        return ns

    def produce_N(self, num_class):
        ns = []
        for i in range(self.num_of_clients):
            if self.is_iid == 1:
                n = self.way + randint(-1, 1)
            else:
                n = self.way
            index = random.sample(range(0, num_class), n)
            ns.append(index)
        return ns

    def dataSetBalanceAllocation(self):
        DataSet = GetDataSet(self.data_set_name, self.is_iid)

        train_data = DataSet.train_data
        train_label = DataSet.train_label
        train_label_value = np.argmax(train_label, axis=1)

        ns = self.produce_N(train_label.shape[1])
        train_index = {}
        test_index = {}
        for i in range(train_label.shape[1]):
            train_index[i] = []
            test_index[i] = []

        for i in range(len(train_label_value)):
            train_index[train_label_value[i]].append(i)

        for i in range(self.num_of_clients):
            k = self.shot + randint(-2, 2)
            train_id = []
            test_id = []
            classes = ns[i]
            for label in classes:
                
                index1 = random.sample(train_index[label], min(k, len(train_index[label])))
                train_id = train_id + index1
                for index in index1:
                    train_index[label].remove(index)
                index2 = random.sample(train_index[label],  min(2*k, len(train_index[label])))
            
                test_id = test_id + index2
                for index in index2:
                    train_index[label].remove(index)

            local_train_data = train_data[train_id]
            local_test_data = train_data[test_id]
            local_train_label = train_label_value[train_id]
            local_test_label = train_label_value[test_id]

            someone = client(torch.tensor(local_train_data), torch.tensor(local_train_label),
                                     torch.tensor(local_test_data), torch.tensor(local_test_label), self.dev, self.net, self.num_in_comm, self.num_of_clients)
            self.clients_set['client{}'.format(i)] = someone


