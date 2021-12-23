# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import pickle
from tqdm import tqdm
import sys
sys.path.append("../")


class Encoder(nn.Module):
    """
     Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size_1, hidden_layer_depth,
                 latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        # Use nn.module from PyTorch
        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size_1,
                                 self.hidden_layer_depth, dropout = dropout,
                                 bidirectional = False) # checking bi-directional mode

            print("Single LSTM model")
            print(self.model)
            print(type(self.model))

        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size_1,
                                self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the LAST HIDDEN
        state of encoder.
        :param x: input to the encoder, of shape
                  (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, shape (batch_size, hidden_size_1)
        """
        # RNN in pytorch takes input x as: (seq_len*batch_size*num_of_features)
        # in contrast to other modules who take it as:
        # (seq_len x batch_size x num_of_features)

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]

        # print("testing Forward")
        # print(h_end.size()) torch.size(32, 512)

        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size_1: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size_1, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size_1 = hidden_size_1
        self.latent_length = latent_length

        # nn.Linear: nodes with linear activation
        self.hidden_to_mean = nn.Linear(self.hidden_size_1, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size_1, self.latent_length)

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
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size_1: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing mean, other log std dev of output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size_1,
                 hidden_layer_depth, latent_length, output_size,
                 dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size_1 = hidden_size_1
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size_1, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size_1, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size_1)
        self.hidden_to_output = nn.Linear(self.hidden_size_1, self.output_size)

        self.decoder_inputs = \
            torch.zeros(self.sequence_length, self.batch_size, 1,
                        requires_grad=True).type(self.dtype)

        self.c_0 = \
            torch.zeros(self.hidden_layer_depth, self.batch_size,
                        self.hidden_size_1, requires_grad=True).type(self.dtype)

        # Xavier weight intilization
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)
        # print("h_state:", h_state.size())

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)

        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class VRAE(BaseEstimator, nn.Module):
    """
    Variational recurrent auto-encoder. This module is used for dimensionality
    reduction of uni or multivariate timeseries.
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
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

    def __init__(self, sequence_length, number_of_features, hidden_size_1=90,
                 hidden_layer_depth=2, latent_length=20, batch_size=32,
                 learning_rate=0.0005, momentum=0.5, n_epochs=10,
                 dropout_rate=0.2, optimizer='Adam', cuda=False, print_every=10,
                 clip=True, max_grad_norm=5, loss='MSELoss',
                 block='LSTM', kl_annealing="CyclicalKL", model_download='.'):

        super(VRAE, self).__init__()

        self.dtype = torch.FloatTensor
        self.use_cuda = cuda and torch.cuda.is_available()
        # use_cuda = args.cuda and torch.cuda.is_available()

        # if not torch.cuda.is_available() and self.use_cuda:
        #     self.use_cuda = False
        #     print("not using cuda")

        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            print("using cuda...")

        # Instantiate encoder
        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size_1=hidden_size_1,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        # Instantiate Lambda compression
        self.lmbd = Lambda(hidden_size_1=hidden_size_1,
                           latent_length=latent_length)

        # Instantiate decoder
        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size_1=hidden_size_1,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        # Set class attributes
        self.sequence_length = sequence_length
        self.hidden_size_1 = hidden_size_1
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.kl_annealing = kl_annealing

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.model_download = model_download

        if self.use_cuda:
            self.cuda()
            print("using cuda...")

        # Set optimizers
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True) # checking amsgrad
            # self.steps = 20
            # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.steps)

        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=self.momentum)
            # self.steps = 20
            # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.steps)
        else:
            raise ValueError('Not a recognized optimizer')

        # Set losses (for reconstruction)
        if loss == 'SmoothL1Loss':
            print("Using SmoothL1Loss")
            self.loss_fn = nn.SmoothL1Loss(size_average=False, reduction='mean') # "sum"
        elif loss == 'MSELoss':
            print("Using MSELoss")
            self.loss_fn = nn.MSELoss(size_average=False, reduction='mean')

    def __repr__(self):
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
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        # kl_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        # Reconstruction loss
        recon_loss = loss_fn(x_decoded, x)

        return recon_loss + (self.kl_weight * kl_loss), recon_loss, self.kl_weight * kl_loss

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
        x = Variable(X[:,:,:].type(self.dtype), requires_grad=requires_grad)

        x_decoded, _ = self(x)
        total_loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)
        # detach() method constructs a new view on a tensor which is declared not to need gradients

        return total_loss, recon_loss, kl_loss, x

    def _train(self, train_loader, test_loader, epoch):
        """
        This method performs batch training per epoch.
        For each epoch, given num_samples and batch_size, it runs a total of num_samples/batch_size
        :param train_loader: data_loader with specifc batch size and shuffling
        :return: epoch losses
        """

        train_epoch_loss = 0
        train_epoch_recon_loss = 0
        train_epoch_kl_loss = 0
        train_epoch_results = []

        test_epoch_results = []

        t = 0 # batch idx

        print()
        print("Current epoch: %s" % epoch)
        for t, X in enumerate(tqdm(train_loader)):

            self.train() # check: inside or outside first for

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes
            # dataloader gives output in (batch_size, seq_len, num_of_features)
            # transform into             (seq_len, batch_size, num_of_features)
            X = X.permute(1,0,2) # torch.Size([200, 32, 6])

            # reset gradients
            self.optimizer.zero_grad()

            # fetch batch losses
            batch_total_loss, batch_recon_loss, batch_kl_loss, _ = self.compute_loss(X, requires_grad=True)

            # compute gradients
            batch_total_loss.backward()

            # clip gradients (avoid explosion)
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               max_norm = self.max_grad_norm)

            # epoch loss accumulator of batch losses
            train_epoch_loss += batch_total_loss.item()
            train_epoch_recon_loss += batch_recon_loss.item()
            train_epoch_kl_loss += batch_kl_loss.item()

            # perform parameter update
            self.optimizer.step()

            # update learning rate
            # self.scheduler.step()

            # print loss values at each batch idx
            if (t+1) % self.print_every == 0:
                # print("Batch losses at idx %d: batch_loss = %.4f, batch_recon_loss = %.4f, batch_kl_loss = %.4f" \
                        # % (t+1, batch_total_loss.item(), batch_recon_loss.item(), batch_kl_loss.item()))

                # train_epoch_results.append({
                #     "train_epoch": epoch,
                #     "train_batch idx": t+1,
                #     "train_batch_total_loss": train_epoch_loss,
                #     "train_batch_recon_loss": train_epoch_recon_loss,
                #     "train_batch_kl_loss": train_epoch_kl_loss
                # })

                # perform testing at every epoch
                # perform forward pass to compute losses
                test_epoch_loss = 0
                test_epoch_recon_loss = 0
                test_epoch_kl_loss = 0

                self.eval()

                with torch.no_grad():
                    for i, Y in enumerate(test_loader):
                        Y = Y[0]
                        Y = Y.permute(1,0,2)

                        batch_total_loss, batch_recon_loss, batch_kl_loss, _ = self.compute_loss(Y, requires_grad=False)
                        test_epoch_loss += batch_total_loss.item()
                        test_epoch_recon_loss += batch_recon_loss.item()
                        test_epoch_kl_loss += batch_kl_loss.item()

                        # test_epoch_results.append({
                        #     "test_epoch": epoch,
                        #     "test_batch idx": i+1,
                        #     "test_batch_total_loss": test_epoch_loss,
                        #     "test_batch_recon_loss": test_epoch_recon_loss,
                        #     "test_batch_kl_loss": test_epoch_kl_loss
                        #     })

                    # print("**********")
                    # print("len test loader", len(test_loader))
                    # print("total steps:", len(test_loader)/self.batch_size)
                    # print("**********")

                    test_epoch_results.append({
                        "test_epoch": epoch,
                        "test_epoch_total_loss": test_epoch_loss / len(test_loader.dataset),
                        "test_epoch_recon_loss": test_epoch_recon_loss / len(test_loader.dataset),
                        "test_epoch_kl_loss": test_epoch_kl_loss / len(test_loader.dataset)
                        })

                    # only printing training losses for now
                    # print()
                    # print("Test average epoch TOTAL loss: {:.4f}".format(test_epoch_loss / len(test_loader)))
                    # print("Test average epoch RECON loss: {:.4f}".format(test_epoch_recon_loss / len(test_loader)))
                    # print("Test average epoch KL loss: {:.4f}".format(test_epoch_kl_loss / len(test_loader)))

        train_epoch_results.append({
            "train_epoch": epoch,
            "train_epoch_total_loss": train_epoch_loss / len(train_loader.dataset), # len(train_loader)/self.batch_size --> len(train_loader) is already training steps
            "train_epoch_recon_loss": train_epoch_recon_loss / len(train_loader.dataset),
            "train_epoch_kl_loss": train_epoch_kl_loss / len(train_loader.dataset)
        })

        print()
        print("Train average epoch TOTAL loss: {:.4f}".format(train_epoch_loss / len(train_loader.dataset)))
        print("Train average epoch RECON loss: {:.4f}".format(train_epoch_recon_loss / len(train_loader.dataset)))
        print("Train average epoch KL loss: {:.4f}".format(train_epoch_kl_loss / len(train_loader.dataset)))

        # print()
        # print("Resetting LR scheduler")
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.steps)

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

        ########################################################################
        # set KL annealing
        self.kl_weights = self._frange_cycle_linear(0.0, 1.0, self.n_epochs, 4) # cyclic
        # beta_np_inc = frange_cycle_linear(0.0, 1.0, self.n_epochs, 1, 0.25) # monotonic
        self.kl_weight = 0

        # if self.kl_annealing == "CyclicalKL":
        #     self.kl_weight = self._cyclical_KL_annealing(epoch)
        # elif self.kl_annealing == "MonotonicalKL":
        #     self.kl_weight = self._monotonic_KL_annealing(epoch)
        # else:
        #     print("No valid annealing method introduced. Training with constatnt KL weight!")
        #     self.kl_weight = 1
        ########################################################################

        print()
        print("**********")
        print("len train dataset", len(train_dataset))
        print("len test dataset", len(test_dataset))
        print()
        print("len train loader", len(train_loader))
        print("len test loader", len(test_loader))
        print()
        print("total train steps:", len(train_dataset)/self.batch_size)
        print("total test steps:", len(test_dataset)/self.batch_size)
        print("**********")
        print()

        print("-----------------")
        print("Started training model, iterating over {} epochs".format(self.n_epochs))
        start = time.time()

        # iterate over all epochs
        for epoch in tqdm(range(self.n_epochs)):

            # update KL weight at every epoch
            self.kl_weight = self.kl_weights[epoch]
            print ("Current KL Weight:" + str(self.kl_weight))

            # batch training
            train_epoch_results, test_epochs_results = self._train(train_loader, test_loader, epoch)

            # append results of epoch
            self.training_results.append(train_epoch_results)
            self.testing_results.append(test_epochs_results)

        elapsed_time_fl = (time.time() - start)
        print("-----------------")
        print("finished training, total ellapsed train time:", elapsed_time_fl)

        print("-----------------")
        self.is_fitted = True
        print("changed flag to fitted...")

    def _frange_cycle_linear(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        """
        Custom function for multiple annealing scheduling. Given number of epochs,
        it returns the value of KL weight at each epoch as a list.

        Obtained from: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
        """
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L

    def _monotonic_KL_annealing(self, epoch):
        """
        Implements monotonic KL annealing
        Based on: https://deepakbaby.github.io/post/vae-insights/)
        Based on: https://github.com/jxhe/vae-lagging-encoder/blob/master/image.py
        """

        kl_start_time =  50 # epoch at which KL is first included
        kl_anneal_time = 30 # number of epochs over which KL scaling is increased from 0 to 1 (warm up)
        # self.anneal_rate = (1.0 - self.kl_weight) / (self.kl_warm_up * len(train_loader))

        if epoch >= kl_start_time:
            self.kl_weight = min(1.0, self.kl_weight + (1./kl_anneal_time))
            print ("Current KL Weight:" + str(self.kl_weight))

        return self.kl_weight

    def _cyclical_KL_annealing(self, epoch):
        """
        Implements cyclic KL annealing
        Based on: https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
        """

        kl_start_time =  50 # epoch at which KL is first included
        kl_anneal_time = 10 # number of epochs over which KL scaling is increased from 0 to 1 (rate)
        kl_cycle = 50 # reset to zero kl weight every i-th cycles

        if epoch >= kl_start_time:
            self.kl_weight = min(1.0, self.kl_weight + (1./kl_anneal_time))
            print ("Current KL weight:" + str(self.kl_weight))

        if epoch % kl_cycle == 0:
            self.kl_weight = 0
            print("Resetting KL weight (completed cycle). KL weight: " + str(self.kl_weight))

        return self.kl_weight

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

    def transform(self, dataset, file_name, save=False, load=False):
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

        test_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False, # don't shuffle for val data
                                 num_workers=2,
                                 drop_last=False) # keep all test data for later labelling
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
        x_decoded, _ = self(x) # check this self

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, file_name, save = False, load = False):
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
                                 shuffle = False, # no shuffle for test_loader
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
                with torch.no_grad():
                    x_decoded = []

                    # similar to _train()
                    for t, x in enumerate(test_loader):
                        x = x[0]
                        # required to swap axes
                        # dataloader outputs (batch_size, seq_len, num_of_features)
                        x = x.permute(1, 0, 2)

                        x_decoded_each = self._batch_reconstruct(x)
                        x_decoded.append(x_decoded_each)

                    print("finished reconstruction...")
                    print("--------------------------")
                    x_decoded = np.concatenate(x_decoded, axis=1)

                    if save:
                        if os.path.exists(self.model_download):
                            pass
                        else:
                            os.mkdir(self.model_download)
                        # x_decoded.dump(self.model_download + '/x_decoded.pkl')
                        x_decoded.dump(file_name)
                        print("saved reconstructed data x_decoded")
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
