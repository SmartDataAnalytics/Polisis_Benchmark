import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from privacy_policies_dataset import PrivacyPoliciesDataset as PPD
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from os.path import join
from math import ceil

class CNN(nn.Module):


    """
    
    Convolutional Neural Model used for training the models. The total number of kernels that will be used in this
    CNN is Co * len(Ks). 
    
    Args:
        weights_matrix: numpy.ndarray, the shape of this n-dimensional array must be (words, dims) were words is
        the number of words in the vocabulary and dims is the dimensionality of the word embeddings.
        Co (number of filters): integer, stands for channels out and it is the number of kernels of the same size that will be used.
        Hu: integer, stands for number of hidden units in the hidden layer.
        C: integer, number of units in the last layer (number of classes)
        Ks: list, list of integers specifying the size of the kernels to be used. 
     
    """
    
    def __init__(self, vocab_size, emb_dim, Co, Hu, C, Ks, name = 'generic'):
        
        super(CNN, self).__init__()
        
        self.num_embeddings = vocab_size
        
        self.embeddings_dim = emb_dim

        self.padding_index = 0
        
        self.cnn_name = 'cnn_' + str(emb_dim) + '_' + str(Co) + '_' + str(Hu) + '_' + str(C) + '_' + str(Ks) + '_' + name

        self.Co = Co
        
        self.Hu = Hu
        
        self.C = C
        
        self.Ks = Ks
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embeddings_dim, self.padding_index)
        # self.embedding = nn.Embedding(self.num_embeddings, self.embeddings_dim)

        self.convolutions = nn.ModuleList([nn.Conv2d(1,self.Co,(k, self.embeddings_dim)) for k in self.Ks])
        # activation function for hidden layers =  Rectified Linear Unit
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.5)
        units = [self.Co * len(self.Ks)] + Hu
        
        self.linear_layers = nn.ModuleList([nn.Linear(units[k],units[k+1]) for k in range(len(units)-1)])
        
        self.linear_last = nn.Linear(self.Hu[-1], self.C)
        
        #self.linear1 = nn.Linear(self.Co * len(self.Ks), self.Hu)
        
        #self.linear2 = nn.Linear(self.Hu, self.C)
        # activation function of output layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        
        #size(N,1,length) to size(N,1,length,dims)
        
        x = self.embedding(x)
        
        #size(N,1,length,dims) to size(N,1,length)
        
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convolutions]
        
        #size(N,1,length) to (N, Co * len(Ks))
        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x,1)
        
        #size(N, Co * len(Ks)) to size(N, Hu_last)
        
        for linear in self.linear_layers:

            x = linear(x)

            x = self.relu(x)
        
        #size(N, Hu_last) to size(N, C)
                  
        x = self.drop_out(x)
        x = self.linear_last(x)

        x = self.sigmoid(x)
        
        return x
    
    def load_pretrained_embeddings(self, weights_matrix):
                
        self.embedding = self.embedding.from_pretrained(torch.tensor(weights_matrix).float())
    
    def save_cnn_params(self):
        
        cnn_params = {'vocab_size': self.num_embeddings,
                      
                      'emb_dim': self.embeddings_dim,
                      
                      'Co': self.Co,
                      
                      'Hu': self.Hu,
                      
                      'C': self.C,
                      
                      'Ks': self.Ks,
                     
                      'name': self.cnn_name}
        
        output_path = join("trained_models", self.cnn_name + "_params.pkl")
        
        with open(output_path, "wb") as output_file:
        
            pickle.dump(cnn_params, output_file)
        
    def train_CNN(self, train_dataset, validation_dataset, lr = 0.01, epochs_num = 100, batch_size = 40, alpha = 0, momentum = 0.9):
        """

        This function trains a CNN model using gradient descent with the posibility of using momentum. 

        Args:
            model: cnn.CNN, an instance of a model of the class cnn.CNN 
            train_dataset: Dataset, Dataset containing the data that will be used for training
            validation_dataset: Dataset, Dataset containing the data that will be used for validating the model
            lr: double, learning rate that we want to use in the learning algorithm
            epochs_num: integer, number of epochs
            momentum: double, momentum paramenter that tunes the momentum gradient descent algorithm    
        Returns:
            epochs: list, list containing all the epochs
            losses: list, list containing the loss at the beginning of each epoch

        """
        threshold = 0.5
        # optimizer = SGD(self.parameters(), lr = lr, weight_decay = alpha, momentum = momentum)
        optimizer = Adam(self.parameters())

        criterion = nn.BCELoss()

        train_losses = []

        validation_losses = []

        f1_scores_validations = []
        precisions_validations = []
        recalls_validations = []

        epochs = []

        start = time.time()

        remaining_time = 0

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = PPD.collate_data)
        best_f1score_validation = 0
        patience = 0
        for epoch in range(epochs_num):

            super(CNN, self).train()

            for i_batch, sample_batched in enumerate(train_dataloader):

                input = sample_batched[0]

                target = sample_batched[1].float()
                #optimizer.zero_grad() clears x.grad for every parameter x in the optimizer. It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
                self.zero_grad()

                output = self(input)

                train_loss = criterion(output, target)
                #loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x. In pseudo-code: x.grad += dloss/dx
                train_loss.backward()
                #optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs: x += -lr * x.grad
                optimizer.step()

            super(CNN, self).eval()

            validation_segments, validation_labels = PPD.collate_data(validation_dataset)

            validation_loss = criterion(self(validation_segments.long()), validation_labels.float())

            f1_scores_validation = self.f1_score(self(validation_segments.long()), validation_labels.float(), threshold)[0]
            precisions_validation = self.f1_score(self(validation_segments.long()), validation_labels.float(), threshold)[1]
            recalls_validation = self.f1_score(self(validation_segments.long()), validation_labels.float(), threshold)[2]

            if (ceil(f1_scores_validation * 100) / 100) <= (ceil(best_f1score_validation * 100) / 100):
                patience = patience + 1
            else:
                best_f1score_validation = f1_scores_validation
                patience = 0


            end = time.time()

            remaining_time = remaining_time * 0.90 + ((end - start) * (epochs_num - epoch + 1) / (epoch + 1)) * 0.1

            remaining_time_corrected = remaining_time / (1 - (0.9 ** (epoch + 1)))

            epoch_str = "last epoch finished: " + str(epoch)

            progress_str = "progress: " + str((epoch + 1) * 100 / epochs_num) + "%"

            time_str = "time: " + str(remaining_time_corrected / 60) + " mins"

            sys.stdout.write("\r" + epoch_str + " -- " + progress_str + " -- " + time_str)

            sys.stdout.flush()

            train_losses.append(train_loss.item())

            validation_losses.append(validation_loss.item())

            f1_scores_validations.append(f1_scores_validation)
            precisions_validations.append(precisions_validation)
            recalls_validations.append(recalls_validation)

            epochs.append(epoch)
            # if patience == 15:
            #     break

        print("\n" + "Training completed. Total training time: " + str(round((end - start) / 60, 2)) + " mins")

        return epochs, train_losses, validation_losses, f1_scores_validations, precisions_validations, recalls_validations
    
    def train_label(self, train_dataset, validation_dataset, label, lr = 0.02, epochs_num = 100, batch_size = 40, alpha = 0, momentum = 0.9):
        """

        This function trains a CNN model using gradient descent with the posibility of using momentum. 

        Args:
            model: cnn.CNN, an instance of a model of the class cnn.CNN 
            train_dataset: Dataset, Dataset containing the data that will be used for training
            validation_dataset: Dataset, Dataset containing the data that will be used for validating the model
            lr: double, learning rate that we want to use in the learning algorithm
            epochs_num: integer, number of epochs
            momentum: double, momentum paramenter that tunes the momentum gradient descent algorithm    
        Returns:
            epochs: list, list containing all the epochs
            losses: list, list containing the loss at the beginning of each epoch

        """

        label_name = list(train_dataset.labels.items())[label][0]
        
        print("Training label {} ... ".format(label_name))
        
        optimizer = SGD(self.parameters(), lr = lr, weight_decay = alpha, momentum = momentum)

        criterion = nn.BCELoss()

        train_losses = []

        validation_losses = []

        epochs = []

        start = time.time()

        remaining_time = 0

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = PPD.collate_data)

        for epoch in range(epochs_num):

            for i_batch, sample_batched in enumerate(train_dataloader):

                input = sample_batched[0]

                target = sample_batched[1][:,label].unsqueeze(1).float()

                self.zero_grad()

                output = self(input)

                train_loss = criterion(output, target)

                train_loss.backward()

                optimizer.step()

            validation_segments, validation_labels = PPD.collate_data(validation_dataset)

            validation_loss = criterion(self(validation_segments.long()), validation_labels[:,label].unsqueeze(1).float())

            end = time.time()

            remaining_time = remaining_time * 0.90 + ((end - start) * (epochs_num - epoch + 1) / (epoch + 1)) * 0.1

            remaining_time_corrected = remaining_time / (1 - (0.9 ** (epoch + 1)))

            epoch_str = "last epoch finished: " + str(epoch)

            progress_str = "progress: " + str((epoch + 1) * 100 / epochs_num) + "%"

            time_str = "time: " + str(remaining_time_corrected / 60) + " mins"

            sys.stdout.write("\r" + epoch_str + " -- " + progress_str + " -- " + time_str)

            sys.stdout.flush()

            train_losses.append(train_loss.item())

            validation_losses.append(validation_loss.item())

            epochs.append(epoch)

        print("\n" + "Training completed. Total training time: " + str(round((end - start) / 60, 2)) + " mins")

        return epochs, train_losses, validation_losses
    
    def train_label_weigthed(self, train_dataset, validation_dataset, label, lr = 0.02, epochs_num = 100, batch_size = 40, alpha = 0, momentum = 0.9):
        """

        This function trains a CNN model using gradient descent with the posibility of using momentum. 

        Args:
            model: cnn.CNN, an instance of a model of the class cnn.CNN 
            train_dataset: Dataset, Dataset containing the data that will be used for training
            validation_dataset: Dataset, Dataset containing the data that will be used for validating the model
            lr: double, learning rate that we want to use in the learning algorithm
            epochs_num: integer, number of epochs
            momentum: double, momentum paramenter that tunes the momentum gradient descent algorithm    
        Returns:
            epochs: list, list containing all the epochs
            losses: list, list containing the loss at the beginning of each epoch

        """
                
        def get_proportions(dataset):
            
            positive_label = dataset.labels_tensor[:,label].sum()
        
            negative_label = (1 - dataset.labels_tensor[:,label]).sum()
            
            total_examples = positive_label + negative_label
            
            imbalance = abs(positive_label - 0.5) > 0.4
        
            if imbalance:
            
                if positive_label < negative_label:
                
                    w_p = 1
                
                    w_n = positive_label / negative_label
                
                else:
                
                    w_p = negative_label / positive_label
                
                    w_n = 1
            
            else:
            
                w_p = w_n = 1
                
            return w_p, w_n
            
        def get_w(labels, w_p, w_n):
            
            positives = labels
            
            negatives = 1 - labels
            
            w = w_p * positives + w_n * negatives
            
            return w
        
#         positive_label = train_dataset.labels_tensor[:,label].sum()
        
#         negative_label = (1 - train_dataset.labels_tensor[:,label]).sum()
        
#         total_examples = positive_label + negative_label
        
#         print('num examples {}'.format(positive_label + negative_label))
        
#         print('% positive labels: {}'.format(positive_label/total_examples))
        
#         print('% negative labels: {}'.format(negative_label/total_examples))
        
#         imbalance = abs(positive_label - 0.5) > 0.4
        
#         if imbalance:
            
#             if positive_label < negative_label:
                
#                 w_p = 1
                
#                 w_n = positive_label / negative_label
                
#             else:
                
#                 w_p = negative_label / positive_label
                
#                 w_n = 1
            
#         else:
            
#             w_p = w_n = 1
            
#         print('w_p: {}'.format(w_p))
        
#         print('w_n: {}'.format(w_n))

        w_p, w_n = get_proportions(train_dataset)        
        
        label_name = train_dataset.labels.items()[label][0]
        
        print("Training label {} ... ".format(label_name))
        
        optimizer = SGD(self.parameters(), lr = lr, weight_decay = alpha, momentum = momentum)

        train_losses = []

        validation_losses = []

        epochs = []

        start = time.time()

        remaining_time = 0

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = PPD.collate_data)
        
        validation_segments, validation_labels = PPD.collate_data(validation_dataset)
        
        weight_matrix_v = get_w(validation_labels[:,label].unsqueeze(1), w_p, w_n)
        
        criterion_v = nn.BCELoss(weight=weight_matrix_v.float())
        
        print('w_p: {} and w_n: {}'.format(w_p, w_n))

        for epoch in range(epochs_num):

            for i_batch, sample_batched in enumerate(train_dataloader):

                input = sample_batched[0]

                target = sample_batched[1][:,label].unsqueeze(1)
                
                weight_matrix = w_p * target + w_n * (1 - target)
                
                criterion = nn.BCELoss(weight=weight_matrix.float())

                self.zero_grad()

                output = self(input)

                train_loss = criterion(output, target.float())

                train_loss.backward()

                optimizer.step()

            validation_loss = criterion_v(self(validation_segments.long()), validation_labels[:,label].unsqueeze(1).float())

            end = time.time()

            remaining_time = remaining_time * 0.90 + ((end - start) * (epochs_num - epoch + 1) / (epoch + 1)) * 0.1

            remaining_time_corrected = remaining_time / (1 - (0.9 ** (epoch + 1)))

            epoch_str = "last epoch finished: " + str(epoch)

            progress_str = "progress: " + str((epoch + 1) * 100 / epochs_num) + "%"

            time_str = "time: " + str(remaining_time_corrected / 60) + " mins"

            sys.stdout.write("\r" + epoch_str + " -- " + progress_str + " -- " + time_str)

            sys.stdout.flush()

            train_losses.append(train_loss.item())

            validation_losses.append(validation_loss.item())

            epochs.append(epoch)

        print("\n" + "Training completed. Total training time: " + str(round((end - start) / 60, 2)) + " mins")

        return epochs, train_losses, validation_losses
    
    def print_results(self, train_dataset, validation_dataset, threshold):

        labels = train_dataset.labels
        
        y_train = train_dataset.labels_tensor

        y_validation = validation_dataset.labels_tensor

        x_train = PPD.collate_data(train_dataset)[0]

        x_validation = PPD.collate_data(validation_dataset)[0]


        y_hat_train = self(x_train)

        y_hat_validation = self(x_validation)


        # This will be the x axis
        threshold_list = np.arange(0.0, 1, 0.01)

        # These will be the y axis data
        f1_scores_validation = [self.f1_score(y_validation, y_hat_validation, t)[0] for t in threshold_list]

        precisions_validation = [self.f1_score(y_validation, y_hat_validation, t)[1] for t in threshold_list]

        recalls_validation = [self.f1_score(y_validation, y_hat_validation, t)[2] for t in threshold_list]

        f1_scores_train = [self.f1_score(y_train, y_hat_train, t)[0] for t in threshold_list]

        precisions_train = [self.f1_score(y_train, y_hat_train, t)[1] for t in threshold_list]

        recalls_train = [self.f1_score(y_train, y_hat_train, t)[2] for t in threshold_list]


        count_train = y_train.sum(0).div(len(y_train))
        
        print("{} Labels T".format(y_train.sum()))
        
        print("{} Segments T".format(len(y_train)))
        
        count_valid = y_validation.sum(0).div(len(y_validation))
              
        print("{} Labels V".format(y_validation.sum()))
        
        print("{} Segments V".format(len(y_validation)))

        """
        Here comes the pyplot code
        """

        fig = plt.figure(figsize=(15,4))

        # We start with the three pyplot axis we want. One for F1, another for precision and one last one for recall
        ax_f1 = fig.add_subplot(131)

        ax_precision = fig.add_subplot(132)

        ax_recall = fig.add_subplot(133)

        # We now plot all the data in te corresponding axis
        ax_f1.plot(threshold_list, f1_scores_validation, label='validation')

        ax_f1.plot(threshold_list, f1_scores_train, label='train')


        ax_f1.set_title('F1 Score vs Threshold')

        ax_f1.set_ylim(0,1.05)

        ax_f1.legend()

        ax_precision.plot(threshold_list, precisions_validation, label='validation')

        ax_precision.plot(threshold_list, precisions_train, label='train')


        ax_precision.set_title('Precision vs Threshold')

        ax_precision.set_ylim(0,1.05)

        ax_precision.legend()

        ax_recall.plot(threshold_list, recalls_validation, label='validation')

        ax_recall.plot(threshold_list, recalls_train, label='train')


        ax_recall.set_title('Recall vs Threshold')

        ax_recall.set_ylim(0,1.05)

        ax_recall.legend()

        plt.show()

        # We show the overall F1, precision and recall for a threshold of 0.5 given by the variable threshold
        
        f1_micro, precision_micro, recall_micro = self.f1_score(y_validation, y_hat_validation, 0.5)
        
        f1_macro, precision_macro, recall_macro = self.f1_score(y_validation, y_hat_validation, 0.5, macro = True)

        print("Scores with " + str(threshold) + " threshold")

        print("-" * 35 * 3)

        print("f1 micro        |" + str(f1_micro))

        print("precision micro |" + str(precision_micro))

        print("recall micro    |" + str(recall_micro))

        print("-" * 35 * 3)

        print("f1 macro        |" + str(f1_macro))

        print("precision macro |" + str(precision_macro))

        print("recall macro    |" + str(recall_macro))

        print("-" * 35 * 3)


        # We show the F1, precision and recall per label for a threshold given by the variable threshold
        scores_list = self.f1_score_per_label(y_validation, y_hat_validation, threshold)

        print("\n" + "Score per label with " + str(threshold) + " threshold")

        print("-" * 35 * 3)

        row_format = "{:<48}" + "{:<10}" * 5

        print(row_format.format("Label", "F1", "Precision", "Recall", "Count T.", "Count V."))

        print("-" * 35 * 3)

        for label, index in labels.items():
            
            f1_label = ceil(scores_list[0][index]*100)/100
            
            precision_label = ceil(scores_list[1][index]*100)/100
            
            recall_label = ceil(scores_list[2][index]*100)/100
            
            ct_label = ceil(count_train[index]*100)/100
            
            cv_label = ceil(count_valid[index]*100)/100
                      
            print(row_format.format(label, f1_label, precision_label, recall_label, ct_label, cv_label))

        # We save the figure into a picture
        fig.savefig(fname = join("trained_models_pics" ,self.cnn_name + '.png'), format = 'png')
        
    def print_results_best_t(self, validation_dataset, best_t):
        
        y_validation = validation_dataset.labels_tensor

        x_validation = PPD.collate_data(validation_dataset)[0]

        y_hat_validation = self(x_validation)
        
        labels = validation_dataset.labels
        
        scores_list = self.f1_score_per_label(y_validation, y_hat_validation, best_t)
        
        row_format = "{:<48}" + "{:<10}" * 3

        print(row_format.format("Label", "F1", "Precision", "Recall"))

        print("-" * 35 * 3)

        for label, index in labels.items():
            #f1_label = scores_list[0][index]

            #precision_label = scores_list[1][index]

            #recall_label = scores_list[2][index]

            f1_label = ceil(scores_list[0][index]*100)/100
            
            precision_label = ceil(scores_list[1][index]*100)/100
            
            recall_label = ceil(scores_list[2][index]*100)/100
                      
            print(row_format.format(label, f1_label, precision_label, recall_label))
            
        f1_mean = torch.mean(scores_list[0]).item()
        
        precision_mean = torch.mean(scores_list[1]).item()
        
        recall_mean = torch.mean(scores_list[2]).item()
        
        print('macro averages')
        
        print('F1: {}'.format(f1_mean))
        
        print('Precision: {}'.format(precision_mean))
        
        print('Recall: {}'.format(recall_mean))
        
    def print_results_label(self, train_dataset, validation_dataset, label, threshold):

        labels = train_dataset.labels
        
        label_name = list(train_dataset.labels.items())[label][0]
        
        y_train = train_dataset.labels_tensor[:,label].unsqueeze(1)

        y_validation = validation_dataset.labels_tensor[:,label].unsqueeze(1)

        x_train = PPD.collate_data(train_dataset)[0]

        x_validation = PPD.collate_data(validation_dataset)[0]

        y_hat_train = self(x_train)

        y_hat_validation = self(x_validation)

        # This will be the x axis
        threshold_list = np.arange(0.0, 1, 0.01)

        # These will be the y axis data
        f1_scores_validation = [self.f1_score(y_validation, y_hat_validation, t)[0] for t in threshold_list]

        precisions_validation = [self.f1_score(y_validation, y_hat_validation, t)[1] for t in threshold_list]

        recalls_validation = [self.f1_score(y_validation, y_hat_validation, t)[2] for t in threshold_list]

        f1_scores_train = [self.f1_score(y_train, y_hat_train, t)[0] for t in threshold_list]

        precisions_train = [self.f1_score(y_train, y_hat_train, t)[1] for t in threshold_list]

        recalls_train = [self.f1_score(y_train, y_hat_train, t)[2] for t in threshold_list]
        
        count_train = y_train.sum(0).div(len(y_train))
        
        print("{} Labels Train".format(y_train.sum()))

        print("{} Segments Train".format(len(y_train)))
        
        count_valid = y_validation.sum(0).div(len(y_validation))
              
        print("{} Labels Validation".format(y_validation.sum()))
        
        print("{} Segments Validation".format(len(y_validation)))

        """
        Here comes the pyplot code
        """

        fig = plt.figure(figsize=(15,4))

        # We start with the three pyplot axis we want. One for F1, another for precision and one last one for recall
        ax_f1 = fig.add_subplot(131)

        ax_precision = fig.add_subplot(132)

        ax_recall = fig.add_subplot(133)

        # We now plot all the data in te corresponding axis
        ax_f1.plot(threshold_list, f1_scores_validation, label='validation')

        ax_f1.plot(threshold_list, f1_scores_train, label='train')

        ax_f1.set_title('F1 Score vs Threshold')

        ax_f1.set_ylim(0,1.05)

        ax_f1.legend()

        ax_precision.plot(threshold_list, precisions_validation, label='validation')

        ax_precision.plot(threshold_list, precisions_train, label='train')

        ax_precision.set_title('Precision vs Threshold')

        ax_precision.set_ylim(0,1.05)

        ax_precision.legend()

        ax_recall.plot(threshold_list, recalls_validation, label='validation')

        ax_recall.plot(threshold_list, recalls_train, label='train')

        ax_recall.set_title('Recall vs Threshold')

        ax_recall.set_ylim(0,1.05)

        ax_recall.legend()

        plt.show()

        # We show the overall F1, precision and recall for a threshold of 0.5 given by the variable threshold
        f1, precision, recall = self.f1_score(y_validation, y_hat_validation, 0.5)

        print("Scores with " + str(threshold) + " threshold")

        #print("-" * 35 * 3)

        print("f1        |" + str(f1))

        print("precision |" + str(precision))

        print("recall    |" + str(recall))

        #print("-" * 35 * 3)

        # We save the figure into a picture
        fig.savefig(fname = join("trained_models_pics" ,self.cnn_name + '.png'), format = 'png')

    @staticmethod
    def get_best_thresholds(y_test, y_hat_test, labels):
        
        threshold_list = np.arange(0.0, 1, 0.01)

        best_f1_label = np.zeros((12))

        best_t_label = np.zeros((12))

        for label, index in labels.items():

            best_f1 = 0

            best_t = 0

            for t in threshold_list:

                current_f1 = CNN.f1_score_per_label(y_test, y_hat_test, t)[0][labels[label]].item()

                if current_f1 > best_f1: 

                    best_f1 = current_f1

                    best_t = t

            best_f1_label[index] = best_f1

            best_t_label[index] = best_t

        return best_f1_label, best_t_label

    @staticmethod
    def get_best_threshold(y_test, y_hat_test):

        threshold_list = np.arange(0.0, 1, 0.01)

        best_f1 = 0
        best_t = 0

        for t in threshold_list:
            current_f1 = CNN.f1_score(y_test, y_hat_test, t)[0]
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_t = t

        return best_f1, best_t


    @staticmethod    
    def f1_score(y_true, y_pred, threshold, macro = False, eps = 1e-9):
        """

        Computes the f1 score resulting from the comparison between y_true and y_pred after using the threshold set.

        Args: 
            y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the 
            record i whereas the j-th column is the label j.
            y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th 
            row is the record i whereas the j-th column is the label j.
            threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned 
            to a record.
            macro: bool, if false we will return the micro average but if true it will return the macro average.
            eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall. 

        Returns:
            f1: double, the resulting mean f1 score of all the labels (it will be a number between 0 and 1)
            precision: double, the resulting mean precision of all the labels (it will be a number between 0 and 1)
            recall: double, the resulting mean recall of all the labels (it will be a number between 0 and 1)

        """

        y_pred = torch.ge(y_pred.float(), threshold).float()

        y_true = y_true.float()

        tp_l = (y_pred * y_true).sum(0).float()

        fp_l = (y_pred * (1 - y_true)).sum(0).float()

        fn_l = ((1 - y_pred) * y_true).sum(0).float()

        precision_label = tp_l.div(tp_l + fp_l + eps)

        recall_label = tp_l.div(tp_l + fn_l + eps)

        if macro:

            f1_macro = torch.mean((precision_label * recall_label).div(precision_label + recall_label + eps) * 2)

            return f1_macro.item(), torch.mean(precision_label).item(), torch.mean(recall_label).item()

        else: 

            tp = tp_l.sum()

            fp = fp_l.sum()

            fn = fn_l.sum()

            precision = tp / (tp + fp + eps)

            recall = tp / (tp + fn + eps)

            f1_micro = (precision * recall).div(precision + recall + eps) * 2

            return f1_micro.item(), precision.item(), recall.item()
    
    @staticmethod
    def f1_score_per_label(y_true, y_pred, threshold, eps=1e-9):
        """

        Computes the f1 score per label resulting from the comparison between y_true and y_pred after using the threshold 
        set.

        Args: 
            y_true: torch.tensor, 2-dimensional torch.tensor containing the true labels per record. The i-th row is the 
            record i whereas the j-th column is the label j.
            y_pred: torch.tensor, 2-dimensional torch.tensor containing the probabilities assigned to each label. The i-th 
            row is the record i whereas the j-th column is the label j.
            threshold: double, number between 0 and 1 that sets the threshold probability for a label to be truly assigned 
            to a record.
            eps: double, it is just a very small value that avoids dividing by 0 when computing the precision and recall.

        Returns:
            f1: list, the resulting f1 score per label (it will be a number between 0 and 1)
            precision: list, the resulting precision per label (it will be a number between 0 and 1)
            recall: list, the resulting recall per label (it will be a number between 0 and 1)

        """
        
        y_pred = torch.ge(y_pred.float(), threshold).float()

        y_true = y_true.float()

        tp_l = (y_pred * y_true).sum(0).float()

        fp_l = (y_pred * (1 - y_true)).sum(0).float()

        fn_l = ((1 - y_pred) * y_true).sum(0).float()

        precision_label = tp_l.div(tp_l + fp_l + eps)

        recall_label = tp_l.div(tp_l + fn_l + eps)

        f1_label = (precision_label * recall_label).div(precision_label + recall_label + eps) * 2

        return f1_label, precision_label, recall_label
