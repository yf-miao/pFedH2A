import numpy
import numpy as np
import gzip
import os
import platform
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import trange
from PIL import Image
from sklearn.preprocessing import LabelBinarizer


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct()
        elif self.name == 'cifar':
            self.cifarDataSetConstruct()
        elif self.name == 'cifar100':
            self.cifar100DataSetConstruct()
        elif self.name == 'miniImagenet':
            self.miniImagenetDatasetConstruct()

    def miniImagenetDatasetConstruct1(self):
        datafile = '../data/miniImagenet.pkl'
        images, labels = np.load(datafile, allow_pickle=True)
        classes = list(set(labels))
        train_label = np.zeros((images.shape[0], len(classes)))
        for i in range(labels.shape[0]):
            train_label[i, labels[i]] = 1
        self.train_data = images
        self.train_label = train_label

    def mnistDataSetConstruct(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.MNIST(
            root='../data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='../data', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, train_data in enumerate(testloader, 0):
            testset.data, testset.targets = train_data

        train_images = trainset.data.cpu().detach().numpy()
        train_label = trainset.targets.cpu().detach().numpy()
        train_labels = numpy.zeros([train_label.shape[0], 10], dtype=numpy.float64)
        for i in range(train_label.shape[0]):
            train_labels[i][train_label[i]] = 1

        test_images = testset.data.cpu().detach().numpy()
        test_label = testset.targets.cpu().detach().numpy()
        test_labels = numpy.zeros([test_label.shape[0], 10], dtype=numpy.float64)
        for i in range(test_label.shape[0]):
            test_labels[i][test_label[i]] = 1

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[1] == 1
        assert test_images.shape[1] == 1
        
        train_images = np.concatenate((train_images, test_images), axis=0)
        train_labels = np.concatenate((train_labels, test_labels), axis=0)
        
        labels = np.argmax(train_labels, axis=1)
        order = np.argsort(labels)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]

    def cifarDataSetConstruct(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, train_data in enumerate(testloader, 0):
            testset.data, testset.targets = train_data

        train_images = trainset.data.cpu().detach().numpy()
        train_label = trainset.targets.cpu().detach().numpy()
        train_labels = numpy.zeros([train_label.shape[0], 10], dtype=numpy.float64)
        for i in range(train_label.shape[0]):
            train_labels[i][train_label[i]] = 1

        test_images = testset.data.cpu().detach().numpy()
        test_label = testset.targets.cpu().detach().numpy()
        test_labels = numpy.zeros([test_label.shape[0], 10], dtype=numpy.float64)
        for i in range(test_label.shape[0]):
            test_labels[i][test_label[i]] = 1

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[1] == 3
        assert test_images.shape[1] == 3
        train_images = np.concatenate((train_images, test_images), axis=0)
        train_labels = np.concatenate((train_labels, test_labels), axis=0)
        
        labels = np.argmax(train_labels, axis=1)
        order = np.argsort(labels)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]

    def cifar100DataSetConstruct(self,):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, train_data in enumerate(testloader, 0):
            testset.data, testset.targets = train_data

        train_images = trainset.data.cpu().detach().numpy()
        train_label = trainset.targets.cpu().detach().numpy()
        train_labels = numpy.zeros([train_label.shape[0], 100], dtype=numpy.float64)
        for i in range(train_label.shape[0]):
            train_labels[i][train_label[i]] = 1

        test_images = testset.data.cpu().detach().numpy()
        test_label = testset.targets.cpu().detach().numpy()
        test_labels = numpy.zeros([test_label.shape[0], 100], dtype=numpy.float64)
        for i in range(test_label.shape[0]):
            test_labels[i][test_label[i]] = 1

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[1] == 3
        assert test_images.shape[1] == 3

        train_images = np.concatenate((train_images, test_images), axis=0)

        train_labels = np.concatenate((train_labels, test_labels), axis=0)

        labels = np.argmax(train_labels, axis=1)
        order = np.argsort(labels)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]
        
    def miniImagenetDatasetConstruct(self):
        train_dir = '../data/miniImagenet/train'
        test_dir = '../data/miniImagenet/test'
        val_dir = '../data/miniImagenet/val'
        
    
        images = []
        labels = []
        lb = LabelBinarizer()

        for label, class_dir in enumerate(sorted(os.listdir(train_dir))):
            class_path = os.path.join(train_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    img = np.array(Image.open(img_path))
                    img = img.transpose(2, 0, 1)
                    images.append(img)
                    labels.append(label)
        
        for label, class_dir in enumerate(sorted(os.listdir(test_dir)), start=len(os.listdir(train_dir))):
            class_path = os.path.join(test_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    img = np.array(Image.open(img_path))
                    img = img.transpose(2, 0, 1)
                    images.append(img)
                    labels.append(label)

        for label, class_dir in enumerate(sorted(os.listdir(val_dir)), start=len(os.listdir(train_dir)) + len(os.listdir(test_dir))):
            class_path = os.path.join(val_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    img = np.array(Image.open(img_path))
                    img = img.transpose(2, 0, 1)
                    images.append(img)
                    labels.append(label) 

        images = np.array(images, dtype=np.float32)
        
        labels = lb.fit_transform(labels)
        
        id = np.argmax(labels, axis=1)
        order = np.argsort(id)
        self.train_data = images[order]
        self.train_label = labels[order]





