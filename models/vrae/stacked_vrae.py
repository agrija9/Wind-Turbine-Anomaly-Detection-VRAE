# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import os
import time
import pickle
from tqdm import tqdm
import sys
sys.path.append("../")


class Encoder(nn.Module):
    """
    Encoder network containing stacked LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size_1: hidden size of the RNN1
    :param hidden_size_1: hidden size of the RNN2
    :param hidden_size_1: hidden size of the RNN3
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(self, number_of_features,
                 hidden_size_1, hidden_size_2, hidden_size_3,
                 hidden_layer_depth, latent_length,
                 dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_layer_depth = hidden_layer_depth # no. stacked rnn layers
        self.latent_length = latent_length

        if block == 'LSTM':
            # self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)

            # self.model = nn.Sequential(OrderedDict([
            #     ("LSTM1", nn.LSTM(self.number_of_features, self.hidden_size_1, 1, dropout=dropout)),
            #     ("LSTM2", nn.LSTM(self.hidden_size_1, self.hidden_size_2, 1, dropout=dropout)),
            #     ("LSTM3", nn.LSTM(self.hidden_size_2, self.hidden_size_3, 1, dropout=dropout))
            #     ]))

            # self.model = nn.Sequential(
            #              nn.LSTM(self.number_of_features, self.hidden_size_1, 1, dropout=dropout),
            #              nn.LSTM(self.hidden_size_1, self.hidden_size_2, 1, dropout=dropout),
            #              nn.LSTM(self.hidden_size_2, self.hidden_size_3, 1, dropout=dropout)
            #              )

            self.layer1 = nn.LSTM(self.number_of_features, self.hidden_size_1, 1, dropout=dropout)
            self.layer2 = nn.LSTM(self.hidden_size_1, self.hidden_size_2, 1, dropout=dropout)
            self.layer3 = nn.LSTM(self.hidden_size_2, self.hidden_size_3, 1, dropout=dropout)

            print(">>>Built stacked LSTM encoder<<<")
            print("layer 1", self.layer1)
            print("layer 2", self.layer2)
            print("layer 3", self.layer3)
            print("--------------------------------")
            # summary(self.model.cuda(), (self.number_of_features, self.hidden_size_1, 1))
            # print(type(self.model))

        elif block == 'GRU':
            # self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)

            # self.model = nn.Sequential(OrderedDict([
            #     ("GRU1", nn.GRU(self.number_of_features, self.hidden_size_1, 1, dropout=dropout)),
            #     ("GRU2", nn.GRU(self.hidden_size_1, self.hidden_size_2, 1, dropout=dropout)),
            #     ("GRU3", nn.GRU(self.hidden_size_2, self.hidden_size_3, 1, dropout=dropout))
            #     ]))

            self.layer1 = nn.GRU(self.number_of_features, self.hidden_size_1, 1, dropout=dropout)
            self.layer2 = nn.GRU(self.hidden_size_1, self.hidden_size_2, 1, dropout=dropout)
            self.layer3 = nn.GRU(self.hidden_size_2, self.hidden_size_3, 1, dropout=dropout)

            print(">>>Built stacked GRU encoder<<<")
            print(self.layer1)
            print(self.layer2)
            print(self.layer3)
            print("--------------------------------")

        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the LAST HIDDEN
        state of encoder.
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder of shape (batch_size, hidden_size_3)
        """
        # RNN in pytorch takes input x as: (seq_len*batch_size*num_of_features)
        # in contrast to other modules who take it as:
        # (seq_len x batch_size x num_of_features)

        output1, (hn1, c1) = self.layer1(x)
        output2, (hn2, c2) = self.layer2(output1)
        output3, (hn3, c3) = self.layer3(output2)

        # print("output1 encoder shape:", output1.size())
        # print("output2 encoder shape:", output2.size())
        # print("output3 encoder shape:", output3.size())

        hn3 = hn3[-1, :, :]

        # _, (h_end, c_end) = self.model(x)
        # h_end = h_end[-1, :, :]

        # print("hn3 shape:", hn3.size())

        return hn3


class Lambda(nn.Module):
    """
    Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of last hidden state of encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size_3, latent_length):
        super(Lambda, self).__init__()

        # retreive hidden size of last LSTM layer
        self.hidden_size_3 = hidden_size_3
        self.latent_length = latent_length

        # nn.Linear: nodes with linear activation
        self.hidden_to_mean = nn.Linear(self.hidden_size_3, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size_3, self.latent_length)

        # standard Xavier initialization
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """
        Given last hidden state of encoder, passes through a linear layer,
        and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        # input encoder output into nn.Linear of Lambda
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    """
    Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size_1: hidden size of the RNN1
    :param hidden_size_2: hidden size of the RNN2
    :param hidden_size_3: hidden size of the RNN3
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing mean, other log std dev of output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """

    def __init__(self, sequence_length, batch_size,
                 hidden_size_1, hidden_size_2, hidden_size_3,
                 hidden_layer_depth, latent_length, output_size,
                 dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            # self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth) # check the 1

            # self.model = nn.Sequential(OrderedDict([
            #     ("LSTM4", nn.LSTM(1, self.hidden_size_3, 1)),
            #     ("LSTM5", nn.LSTM(self.hidden_size_3, self.hidden_size_2, 1)),
            #     ("LSTM6", nn.LSTM(self.hidden_size_2, self.hidden_size_1, 1))
            #     ]))

            # hidden sizes inverted
            self.layer3 = nn.LSTM(1, self.hidden_size_3, 1)
            self.layer2 = nn.LSTM(self.hidden_size_3, self.hidden_size_2, 1)
            self.layer1 = nn.LSTM(self.hidden_size_2, self.hidden_size_1, 1)

            print(">>>Built stacked LSTM decoder<<<")
            print("layer 3", self.layer3)
            print("layer 2", self.layer2)
            print("layer 1", self.layer1)
            print("--------------------------------")

        elif block == 'GRU':
            # self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)

            # self.model = nn.Sequential(OrderedDict([
            #     ("GRU4", nn.GRU(1, self.hidden_size_3, 1)),
            #     ("GRU5", nn.GRU(self.hidden_size_3, self.hidden_size_2, 1)),
            #     ("GRU6", nn.GRU(self.hidden_size_2, self.hidden_size_1, 1))
            #     ]))

            self.layer3 = nn.GRU(1, self.hidden_size_3, 1)
            self.layer2 = nn.GRU(self.hidden_size_3, self.hidden_size_2, 1)
            self.layer1 = nn.GRU(self.hidden_size_2, self.hidden_size_1, 1)

            print(">>>Built stacked GRU decoder<<<")
            print(self.layer3)
            print(self.layer2)
            print(self.layer1)
            print("--------------------------------")

        else:
            raise NotImplementedError

        ########################################################################
        # TODO: finish decoder

    #     self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size_3)
    #
    #     self.hidden_to_output = nn.Linear(self.hidden_size_1, self.output_size)
    #
    #     self.decoder_inputs = \
    #         torch.zeros(self.sequence_length, self.batch_size, 1,
    #                     requires_grad=True).type(self.dtype)
    #
    #     self.c_0 = \
    #         torch.zeros(self.hidden_layer_depth, self.batch_size,
    #                     self.hidden_size_1, requires_grad=True).type(self.dtype)
    #
    #     # Xavier weight intilization
    #     nn.init.xavier_uniform_(self.latent_to_hidden.weight)
    #     nn.init.xavier_uniform_(self.hidden_to_output.weight)
    #
    # def forward(self, latent):
    #     """
    #     Converts latent to hidden to output
    #     :param latent: latent vector
    #     :return: outputs consisting of mean and std dev of vector
    #     """
    #     h_state = self.latent_to_hidden(latent)
    #
    #     if isinstance(self.layer3, nn.LSTM):
    #         h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
    #         decoder_output, _ = self.layer1(self.decoder_inputs, (h_0, self.c_0))
    #
    #     elif isinstance(self.layer3, nn.GRU):
    #         h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
    #         decoder_output, _ = self.layer1(self.decoder_inputs, h_0)
    #
    #     else:
    #         raise NotImplementedError
    #
    #     out = self.hidden_to_output(decoder_output)
    #
    #     return out
        ########################################################################

        self.latent_to_hidden3 = nn.Linear(self.latent_length, self.hidden_size_3)
        # self.hidden3_to_hidden2 = nn.Linear(self.hidden_size_3, self.hidden_size_2)
        # self.hidden2_to_hidden1 = nn.Linear(self.hidden_size_2, self.hidden_size_1)
        self.hidden1_to_output = nn.Linear(self.hidden_size_1, self.output_size)

        self.decoder_inputs = \
            torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)

        self.c_0 = \
            torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size_3, requires_grad=True).type(self.dtype)

        # Xavier weight intilization
        nn.init.xavier_uniform_(self.latent_to_hidden3.weight)
        # nn.init.xavier_uniform_(self.hidden3_to_hidden2.weight)
        # nn.init.xavier_uniform_(self.hidden2_to_hidden1.weight)
        nn.init.xavier_uniform_(self.hidden1_to_output.weight)

    def forward(self, latent):
        """
        Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        # print("-----------------------------")
        # print("latent shape:", latent.size())
        # print("decoder inputs:", self.decoder_inputs.size())

        h0_state = self.latent_to_hidden3(latent)

        if isinstance(self.layer3, nn.LSTM):
            h_0 = torch.stack([h0_state for _ in range(self.hidden_layer_depth)])
            # print("h_0 decoder shape:", h_0.size())
            # print("c_0 decoder shape:", self.c_0.size())

            output3, (h3, c3) = self.layer3(self.decoder_inputs, (h_0, self.c_0))
            # print("output3 decoder:", output3.size())
            # print("h3 decoder:", h3.size())
            output2, (h2, c2) = self.layer2(output3)
            # print("output2 decoder:", output2.size())
            output1, (h1, c1) = self.layer1(output2) # decoder output
            # print("output1 decoder:", output1.size())
            # decoder_output, _ = self.layer1(self.decoder_inputs, (h_0, self.c_0))

        elif isinstance(self.layer3, nn.GRU):
            h_0 = torch.stack([h0_state for _ in range(self.hidden_layer_depth)])
            ouput3, (h3, c3) = self.layer3(self.decoder_inputs, (h_0, self.c_0))
            output2, (h2, c2) = self.layer2(output3, (h3, c3))
            output1, (h1, c1) = self.layer1(output2, (h2, c2)) # decoder output
            # decoder_output, _ = self.layer1(self.decoder_inputs, (h_0, self.c_0))

        else:
            raise NotImplementedError

        out = self.hidden1_to_output(output1)

        return out
        ########################################################################

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class MultiStackedVRAE(BaseEstimator, nn.Module):
    """
    Variational recurrent auto-encoder. This module is used for dimensionality
    reduction of uni or multivariate timeseries.
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size_1: hidden size of the RNN1
    :param hidden_size_2: hidden size of the RNN2
    :param hidden_size_3: hidden size of the RNN3
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param momentum: indicate momentum to avoid stuck in local minima
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss (inherit from `_Loss` class)
    :param boolean cuda: to be run on GPU or not
    :param print_every: number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param model_download: Download directory where models are to be dumped
    """

    def __init__(self, sequence_length, number_of_features,
                 hidden_size_1=2048, hidden_size_2=512, hidden_size_3=128,
                 hidden_layer_depth=1, latent_length=32, batch_size=32,
                 learning_rate=0.0005, momentum=0.5, n_epochs=10,
                 dropout_rate=0.2, optimizer='Adam', cuda=False, print_every=10,
                 clip=True, max_grad_norm=5, loss='MSELoss',
                 block='LSTM', model_download='.'):

        super(MultiStackedVRAE, self).__init__()

        self.dtype = torch.FloatTensor
        self.use_cuda = cuda and torch.cuda.is_available()
        # use_cuda = args.cuda and torch.cuda.is_available()

        # if not torch.cuda.is_available() and self.use_cuda:
        #     self.use_cuda = False
        #     print("not using cuda")

        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            # print("using cuda...")

        # Instantiate encoder
        self.encoder = Encoder(number_of_features=number_of_features,
                               hidden_size_1=hidden_size_1,
                               hidden_size_2=hidden_size_2,
                               hidden_size_3=hidden_size_3,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        # Instantiate Lambda compression
        self.lmbd = Lambda(hidden_size_3=hidden_size_3,
                           latent_length=latent_length)

        # Instantiate decoder
        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size=batch_size,
                               hidden_size_1=hidden_size_1,
                               hidden_size_2=hidden_size_2,
                               hidden_size_3=hidden_size_3,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        # Set class attributes
        self.sequence_length = sequence_length
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.model_download = model_download

        print(">>>Hidden layer dimensions<<<")
        print("hidden 1:", self.hidden_size_1)
        print("hidden 2:", self.hidden_size_2)
        print("hidden 3:", self.hidden_size_3)

        if self.use_cuda:
            self.cuda()
            # print("using cuda...")

        # Set optimizers
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=learning_rate, momentum=self.momentum)
        else:
            raise ValueError('Not a recognized optimizer')

        # Set losses (for reconstruction)
        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def __repr__(self):
        # return object representation
        return """
        VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves the following pass
        inputs -> encoder -> lambda -> decoder
        :param x: input tensor
        :return: the decoded output and latent vector
        """

        # print("testing forward mehtod:", type(x))
        # print("testing forward mehtod:", x.size())
        # print("testing forward mehtod:", x)

        cell_output = self.encoder(x) # encode
        latent = self.lmbd(cell_output) # lambda compression
        x_decoded = self.decoder(latent) # decode

        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the
        specified loss function
        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """

        # compute mean and variance of latent vectors
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        # KL divergence
        kl_loss = \
        -0.5*torch.mean(1+latent_logvar-latent_mean.pow(2)-latent_logvar.exp())

        # Reconstruction loss
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X, requires_grad=False):
        """
        Given input tensor:
        foward propagate, compute the loss, and backward propagate.
        Represents the LIFECYCLE of a SINGLE ITERATION
        :param X: Input tensor
        :return:
        total loss, reconstruction loss, kl-divergence loss and original input
        """
        # set pytorch Variable for auto-differentiation
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = requires_grad)

        x_decoded, _ = self(x)

        total_loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)
        # detach() method constructs a new view on a tensor which is declared not to need gradients

        return total_loss, recon_loss, kl_loss, x

################################################################################
    # def _train(self, train_loader, epoch):
    #     """
    #     This method trains a single epoch.
    #     For each epoch, given the batch_size, runs total_data/ batch_size
    #     :param train_loader: data_loader with specifc batch size and shuffling
    #     :return:
    #     """
    #     self.train()
    #
    #     train_epoch_loss = 0
    #     train_epoch_recon_loss = 0
    #     train_epoch_kl_loss = 0
    #     train_epoch_results = []
    #
    #     t = 0 # batch idx
    #
    #     print('Epoch: %s' % epoch)
    #     print("no threads using:", torch.get_num_threads())
    #     for t, X in enumerate(tqdm(train_loader)):
    #         print("entered enumerate train_loader...")
    #
    #         # Index first element of array to return tensor
    #         X = X[0]
    #
    #         # required to swap axes
    #         # dataloader gives output in (batch_size*seq_len*num_of_features)
    #         X = X.permute(1,0,2)
    #
    #         # reset gradients
    #         self.optimizer.zero_grad()
    #
    #         # fetch losses
    #         total_loss, recon_loss, kl_loss, _ = self.compute_loss(X)
    #
    #         # compute gradients
    #         total_loss.backward()
    #
    #         # clip gradients (avoid explosion)
    #         if self.clip:
    #             torch.nn.utils.clip_grad_norm_(self.parameters(),
    #                                            max_norm = self.max_grad_norm)
    #
    #         # epoch loss accumulator
    #         train_epoch_loss += total_loss.item()
    #         train_epoch_recon_loss += recon_loss.item()
    #         train_epoch_kl_loss += kl_loss.item()
    #
    #         # perform parameter update
    #         self.optimizer.step()
    #
    #         # print loss values at each batch idx
    #         if (t+1) % self.print_every == 0:
    #             print('Batch idx %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' \
    #                     % (t+1, total_loss.item(), recon_loss.item(), kl_loss.item()))
    #
    #             train_epoch_results.append({
    #                 "train_epoch": epoch,
    #                 "train_batch idx": t+1,
    #                 "train_batch_total_loss": total_loss.item(),
    #                 "train_batch_recon_loss": recon_loss.item(),
    #                 "train_batch_kl_loss": kl_loss.item()
    #             })
    #
    #     print('Train avg. epoch total loss: {:.4f}'.format(train_epoch_loss/t))
    #     print('Train avg. epoch recon loss: {:.4f}'.format(train_epoch_recon_loss/t))
    #     print('Train avg. epoch kl loss: {:.4f}'.format(train_epoch_kl_loss/t))
    #
    #     train_epoch_results.append({
    #         "epoch": train_epoch,
    #         "epoch_total_loss": train_epoch_loss/t,
    #         "epoch_recon_loss": train_epoch_recon_loss/t,
    #         "epoch_kl_loss": train_epoch_kl_loss/t
    #     })
    #
    #     return train_epoch_results
    #
    # def fit(self, dataset):
    #     """
    #     This method runs for all epochs.
    #     Calls `_train` function over a fixed number of epochs (`n_epochs`)
    #     :param dataset: `Dataset` object
    #     :param bool save: If true, dumps the trained model parameters as
    #                       pickle file at `model_download` directory
    #     :return:
    #     """
    #
    #     self.training_results = []
    #
    #     # create pytorch dataloader
    #     num_workers = 0
    #     print("num_workers:", num_workers)
    #     train_loader = DataLoader(dataset = dataset,
    #                               batch_size = self.batch_size,
    #                               shuffle = True,
    #                               num_workers = num_workers,
    #                               drop_last = True)
    #
    #     print("-----------------")
    #     print("started training, iterating over epochs...")
    #     start = time.time()
    #
    #     # Iterate over all epochs
    #     for epoch in tqdm(range(self.n_epochs)):
    #         # send batches of data to train n epochs
    #         # self._train(train_loader)
    #         epoch_results = self._train(train_loader, epoch)
    #         self.training_results.append(epoch_results)
    #
    #     elapsed_time_fl = (time.time() - start)
    #     print("-----------------")
    #     print("finished training, total ellapsed train time:", elapsed_time_fl)
    #
    #     print("-----------------")
    #     self.is_fitted = True
    #     print("changed flag to fitted...")
    #
################################################################################
    def _train(self, train_loader, test_loader, epoch):
        """
        This method trains a single epoch.
        For each epoch, given the batch_size, runs total_data/ batch_size
        :param train_loader: data_loader with specifc batch size and shuffling
        :return:
        """

        train_epoch_loss = 0
        train_epoch_recon_loss = 0
        train_epoch_kl_loss = 0
        train_epoch_results = []

        test_epoch_results = []

        t = 0 # batch idx

        print('Epoch: %s' % epoch)
        print("no threads using:", torch.get_num_threads())
        for t, X in enumerate(tqdm(train_loader)):
            print("entered enumerate train_loader...")

            self.train()

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes
            # dataloader gives output in (batch_size*seq_len*num_of_features)
            X = X.permute(1,0,2)

            # reset gradients
            self.optimizer.zero_grad()

            # fetch losses
            total_loss, recon_loss, kl_loss, _ = self.compute_loss(X, requires_grad=True)

            # compute gradients
            total_loss.backward()

            # clip gradients (avoid explosion)
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               max_norm = self.max_grad_norm)

            # epoch loss accumulator
            train_epoch_loss += total_loss.item()
            train_epoch_recon_loss += recon_loss.item()
            train_epoch_kl_loss += kl_loss.item()

            # perform parameter update
            self.optimizer.step()

            # print loss values at each batch idx
            if (t+1) % self.print_every == 0:
                print('Batch idx %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' \
                        % (t+1, total_loss.item(), recon_loss.item(), kl_loss.item()))

                train_epoch_results.append({
                    "train_epoch": epoch,
                    "train_batch idx": t+1,
                    "train_batch_total_loss": train_epoch_loss,
                    "train_batch_recon_loss": train_epoch_recon_loss,
                    "train_batch_kl_loss": train_epoch_kl_loss
                })

                # perform testing each print_every iterations
                # perform forward pass to compute losses
                test_epoch_loss = 0
                test_epoch_recon_loss = 0
                test_epoch_kl_loss = 0

                self.eval()

                with torch.no_grad():
                    for i, Y in enumerate(test_loader):
                        Y = Y[0]
                        Y = Y.permute(1,0,2)

                        total_loss, recon_loss, kl_loss, _ = self.compute_loss(Y, requires_grad=False)
                        test_epoch_loss += total_loss.item()
                        test_epoch_recon_loss += recon_loss.item()
                        test_epoch_kl_loss += kl_loss.item()

                        test_epoch_results.append({
                            "test_epoch": epoch,
                            "test_batch idx": i+1,
                            "test_batch_total_loss": test_epoch_loss,
                            "test_batch_recon_loss": test_epoch_recon_loss,
                            "test_batch_kl_loss": test_epoch_kl_loss
                            })

                    test_epoch_results.append({
                        "test_epoch": epoch,
                        "test_epoch_total_loss": test_epoch_loss/i,
                        "test_epoch_recon_loss": test_epoch_recon_loss/i,
                        "test_epoch_kl_loss": test_epoch_kl_loss/i
                        })

                    print('Test avg. epoch total loss: {:.4f}'.format(test_epoch_loss/i))
                    print('Test avg. epoch recon loss: {:.4f}'.format(test_epoch_recon_loss/i))
                    print('Test avg. epoch kl loss: {:.4f}'.format(test_epoch_kl_loss/i))

        print('Train avg. epoch total loss: {:.4f}'.format(train_epoch_loss/t))
        print('Train avg. epoch recon loss: {:.4f}'.format(train_epoch_recon_loss/t))
        print('Train avg. epoch kl loss: {:.4f}'.format(train_epoch_kl_loss/t))

        train_epoch_results.append({
            "train_epoch": epoch,
            "train_epoch_total_loss": train_epoch_loss/t,
            "train_epoch_recon_loss": train_epoch_recon_loss/t,
            "train_epoch_kl_loss": train_epoch_kl_loss/t
        })

        return train_epoch_results, test_epoch_results

    def fit(self, train_dataset, test_dataset):
        """
        This method runs for all epochs.
        Calls `_train` function over a fixed number of epochs (`n_epochs`)
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as
                          pickle file at `model_download` directory
        :return:
        """

        self.training_results = []
        self.testing_results = []

        # create pytorch dataloader
        num_workers = 0
        print("num_workers:", num_workers)
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  num_workers = num_workers,
                                  drop_last = True)

        test_loader = DataLoader(dataset = test_dataset,
                                 batch_size = self.batch_size,
                                 shuffle = True,
                                 num_workers = num_workers,
                                 drop_last = True)

        print("-----------------")
        print("started training, iterating over epochs...")
        start = time.time()

        # Iterate over all epochs (send batches of data to train n epochs)
        for epoch in tqdm(range(self.n_epochs)):
            # self._train(train_loader)
            train_epoch_results, test_epochs_results = self._train(train_loader, test_loader, epoch)
            self.training_results.append(train_epoch_results)
            self.testing_results.append(test_epochs_results)

        elapsed_time_fl = (time.time() - start)
        print("-----------------")
        print("finished training, total ellapsed train time:", elapsed_time_fl)

        print("-----------------")
        self.is_fitted = True
        print("changed flag to fitted...")


    """
    The following functions perform a transform (compression) and reconstruction
    on the given tensor x.
    """

    def _batch_transform(self, x):
        """
        Passes the given input tensor into ENCODER and then into LAMBDA function
        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def transform(self, dataset, file_name, save = False, load = False):
        """
        Given input dataset,
        creates dataloader and performs `_batch_transform` on it
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: dumps the latent vector dataframe as a pickle file
        :return:
        """

        self.file_name = file_name
        self.eval() # in evaluation script we don't want to do backprop

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False, # don't shuffle for val data
                                 num_workers = 2,
                                 drop_last=True)
        if load:
            print("loading z_run pickle...")
            with open(self.file_name, 'rb') as file:
                try:
                    while True:
                        return pickle.load(file)
                        # yield pickle.load(file)
                except EOFError:
                    pass

        else:
            if self.is_fitted:
                # dont want to create dynamic graph for these tensors
                with torch.no_grad():
                    z_run = []
                    # print("entered torch no grad")
                    print("-------------------------")
                    print("performing dimension reduction on vectors...")
                    for t, x in enumerate(tqdm(test_loader)):
                    # note that batch_size must be smaller than dataset ALWAYS
                        x = x[0]

                        # print(type(x))
                        print("batch length", len(x))
                        # print("batch shape:", np.shape(x))

                        # swap axes of input arrays
                        # dataloader gives them as:
                        # (batch_size x seq_len x num_of_features)
                        # after permutation:
                        # (seq_len x batch_size x num_of_features)
                        x = x.permute(1, 0, 2)

                        # perform transformation into latent space
                        z_run_each = self._batch_transform(x)
                        z_run.append(z_run_each)

                    print("finished transformation...")
                    print("--------------------------")
                    z_run = np.concatenate(z_run, axis=0)
                    print("latent z_run vectors shape", z_run.shape)

                    if save:
                        if os.path.exists(self.model_download):
                            pass
                        else:
                            os.mkdir(self.model_download)
                        # dump encoded vectors as pickle file
                        z_run.dump(self.file_name)
                        # z_run.dump(self.model_download + '/z_run.pkl')
                        print("saved encoded vectors in:", self.file_name)
                    print("type z_run", type(z_run))
                    return z_run

            # raise exception if model is not fitted
            raise RuntimeError('Model needs to be fit')

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function
        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset,
        creates dataloader, performs `_batch_reconstruct` on it
        Pre-requisite is that model has to be fit
        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 num_workers = 2,
                                 drop_last=True) # no shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                # similar to _train()
                for t, x in enumerate(test_loader):
                    x = x[0]
                    # required to swap axes
                    # dataloader putputs (batch_size*seq_len*num_of_features)
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.model_download):
                        pass
                    else:
                        os.mkdir(self.model_download)
                    x_decoded.dump(self.model_download + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above
        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    # Save model parameters
    def save(self, PATH):
    # def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,
                          `model_download` serves as the download directory
        :return: None
        """
        # PATH = self.model_download + '/' + file_name
        print("------------------------")
        if os.path.exists(self.model_download):
            pass
        else:
            os.mkdir(self.model_download)

        torch.save(self.state_dict(), PATH)
        print("[INFO] Saved model params to {}".format(PATH))

    def save_log(self, TRAIN_pkl, TEST_pkl):
        """
        Save model training values: epoch losses.
        """
        with open(TRAIN_pkl, "wb") as f:
            pickle.dump(self.training_results, f)
            print("[INFO] Train results (losses) saved to {}".format(TRAIN_pkl))

        with open(TEST_pkl, "wb") as f:
            pickle.dump(self.testing_results, f)
            print("[INFO] Test results (losses) saved to {}".format(TEST_pkl))

    # Load model params if it is fitted
    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        print("looking for model in folder:", PATH)
        print("model is fitted, loading model...")
        self.load_state_dict(torch.load(PATH))
        print("loaded model correctly!")
