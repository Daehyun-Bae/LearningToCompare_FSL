# -------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# -------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=10)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-exp", "--exp_date", type=str, default="200325-1718")
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
EXP_DATE = args.exp_date


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size * 3 * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.sigmoid(self.fc2(out))        # Deprecated
        out = torch.sigmoid(self.fc2(out))
        return out


def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders, metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_fn = os.path.join('./models', "{}_miniimagenet_feature_encoder_{}way_{}shot.pkl".format(
        EXP_DATE, CLASS_NUM, SAMPLE_NUM_PER_CLASS))
    relation_network_fn = os.path.join('./models', "{}_miniimagenet_relation_network_{}way_{}shot.pkl".format(
        EXP_DATE, CLASS_NUM, SAMPLE_NUM_PER_CLASS))

    if os.path.exists(feature_encoder_fn):
        feature_encoder.load_state_dict(torch.load(feature_encoder_fn))
        print("load feature encoder success")
    if os.path.exists(relation_network_fn):
        relation_network.load_state_dict(torch.load(relation_network_fn))
        print("load relation network success")

    total_accuracy = 0.0
    for episode in range(EPISODE):
        # test
        print("Episode-{}\tTesting...".format(episode), end='\t')

        accuracies = []
        for i in range(TEST_EPISODE):
            total_rewards = 0
            task = tg.MiniImagenetTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, 15)   # test_dir, 5, 5, 15(test_num))
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                 split="train", shuffle=False)
            num_per_class = 5
            test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=num_per_class, split="test",
                                                               shuffle=False)

            sample_images, sample_labels = sample_dataloader.__iter__().next()
            for test_images, test_labels in test_dataloader:
                batch_size = test_labels.shape[0]
                print(test_labels)
                # print(test_images.size())
                # calculate features
                sample_features = feature_encoder(Variable(sample_images).cuda(GPU))        # [25, 64, 19, 19]
                # print('sample feature(1): ', sample_features.size())
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)    # [5, 5, 64, 19, 19]
                # print('sample feature(2): ', sample_features.size())
                sample_features = torch.sum(sample_features, 1).squeeze(1)                          # [5, 64, 19, 19]
                # print('sample feature(3): ', sample_features.size())
                test_features = feature_encoder(Variable(test_images).cuda(GPU))           # [25, 64, 19, 19]
                # print('test_features(1): ', test_features.size())

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)   # [25, 5, 64, 19, 19]
                # print('sample feature(4): ', sample_features_ext.size())

                test_features_ext = test_features.unsqueeze(0)                                      # [1, 25, 64, 19, 19]
                # print('test_features(2): ', test_features_ext.size())
                test_features_ext = test_features_ext.repeat(1 * CLASS_NUM, 1, 1, 1, 1)             # [5, 25, 64, 19, 19]
                # print('test_features(3): ', test_features_ext.size())
                test_features_ext = torch.transpose(test_features_ext, 0, 1)                        # [25, 5, 64, 19, 19]
                # print('test_features(4): ', test_features_ext.size())

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2)             # [25, 5, 128, 19, 19]
                # print('relation_pairs(1): ', relation_pairs.size())
                relation_pairs = relation_pairs.view(-1, FEATURE_DIM * 2, 19, 19)                   # [125, 128, 19, 19]
                # print('relation_pairs(2): ', relation_pairs.size())
                relations = relation_network(relation_pairs)                                        # [125, 1]
                # print('relations(1): ', relations.size())
                relations = relations.view(-1, CLASS_NUM)                                           # [25, 5]
                # print('relations(2): ', relations.size())
                print(relations.data)
                rel_score = relations.detach().cpu().numpy()
                print(rel_score)
                _, predict_labels = torch.max(relations.data, 1)
                print(predict_labels)
                exit()

                test_labels = test_labels.cuda(GPU)
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)

            accuracy = total_rewards / 1.0 / CLASS_NUM / 15
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)

        print("test accuracy:{:.4f}\th:{:.3f}".format(test_accuracy, h))

        total_accuracy += test_accuracy

    print("aver_accuracy: {:.4f}".format(total_accuracy / EPISODE))


if __name__ == '__main__':
    main()
