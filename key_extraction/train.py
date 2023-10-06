from train_functions import get_loss_function, get_optimizer, compute_accuracy
from data_loader import AudioDataset
import torch.nn.parallel
import torch.optim
import torch
import os
import torch.utils.data
from models import VGG16
import datetime
import sys
import numpy as np


# train function
def train(file, net, train_loader, val_loader, optimizer, cost_function, n_classes, n_clips=5, batch_size=32,
          loss_weight=1, training_iterations=2000, device="cuda:0"):
    top_accuracy = 0
    data_loader_source = iter(train_loader)

    optimizer.zero_grad()  # reset the optimizer gradient
    for iteration in range(training_iterations):
        # this snippet is used because we reason in iterations and if we finish the dataset we need to start again
        try:
            data_source = next(data_loader_source)
        except StopIteration:
            data_loader_source = iter(train_loader)
            data_source = next(data_loader_source)

        # IMPORTANT
        # data are in the shape rows -> item of the batch, columns -> clips, 3rd dim -> classes features
        label = data_source['label'].to(device)
        for clip in range(n_clips):
            inputs = {}

            inputs['RGB'] = data_source['RGB'][:, clip].to(device)
            inputs['EMG'] = data_source['EMG'][:, clip].to(device)

            logits = net.forward(inputs)  # get predictions from the net
            # compute the loss and divide for the number of clips in order to get the average for clip
            loss = cost_function(logits, label) / n_clips
            loss.backward()  # apply the backward

        optimizer.step()  # update the parameters
        optimizer.zero_grad()  # reset gradient of the optimizer for next iteration

        test_metrics = validate(net, val_loader, n_classes, n_clips, batch_size)

        file.write('[{}/{}] ITERATION COMPLETED\n'.format(iteration, training_iterations))
        '''
        file.write('TRAIN: acc@top1={:.2f}%  acc@top5={:.2f}% loss={:.2f}\n',
                   train_metrics['top1'], train_metrics['top5'], loss.item())
        '''
        file.write('TEST: acc@top1={:.2f}%  acc@top5={:.2f}%\n\n'.format(test_metrics['top1']*100, 0))

        if test_metrics['top1'] >= top_accuracy:
            top_accuracy = test_metrics['top1']
            print('ITERATION:' + str(iteration) + ' - BEST ACCURACY: {:.2f}'.format(top_accuracy * 100))
            print(test_metrics['classes_acc'])
        if iteration % 10 == 0:
            print('ITERATION:' + str(iteration))

    file.write('TOP ACCURACY {:.2f}'.format(top_accuracy))

    return top_accuracy


# validation function
def validate(net, val_loader, n_classes, n_clips=5, batch_size=32, device="cuda:0"):

    net.train(False)  # set model to validate

    total_size = len(val_loader.dataset)
    top1_acc = 0
    top5_acc = 0
    val_correct = 0
    counter = {}

    for i in range(n_classes):
        counter[i] = [0, 0]

    with torch.no_grad():  # do not update the gradient
        for iteration, (data_source) in enumerate(val_loader):  # extract batches from the val_loader
            size = data_source['label'].shape[0]
            label = data_source['label'].to(device)  # send label to gpu

            # create a zero array with logits shape
            # rows -> clips, columns -> item of the batch, 3rd dim -> classes prob
            logits = torch.zeros((n_clips, batch_size, n_classes)).to(device)

            inputs = {}
            for clip in range(n_clips):
                # send all the data from the batch related to the given clip
                # inputs is a dictionary with key -> modality, value -> n rows related to the same clip
                inputs['RGB'] = data_source['RGB'][:, clip].to(device)
                inputs['EMG'] = data_source['EMG'][:, clip].to(device)

                output = net(inputs)  # get predictions from the net
                logits[clip] = output  # save them in the row related to the clip in logits

            # perform mean over the rows to obtain avg predictions for each class between the several clips
            logits = torch.mean(logits, dim=0)

            _, predicted = torch.max(logits.data, 1)
            for i in range(size):
                if predicted[i] == data_source['label'][i].item():
                    counter[data_source['label'][i].item()][0] += 1
                counter[data_source['label'][i].item()][1] += 1

            val_correct += (predicted == label).sum().item()

    # compute the accuracy
    accuracy = val_correct/total_size
    for i in range(n_classes):
        if counter[i][1] != 0:
            counter[i] = counter[i][0]/counter[i][1]
        else:
            counter[i] = 0
    test_results = {'top1': accuracy,'classes_acc':counter}

    return test_results


def main():
    device = "cuda:0"

    lr = 0.001
    wd = 1e-7
    momentum = 0.9
    loss_weight = 1

    batch_size = 32
    num_frames = 16

    log_path = 'logs'

    dataset = sys.argv[1]
    split = sys.argv[2]
    n_clips = int(sys.argv[3])
    n_classes = int(sys.argv[4])
    annotations_path = sys.argv[5]
    features_path = sys.argv[6]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = dataset + '_' + timestamp + '.txt'

    file = open(os.path.join(log_path, file_name), 'w')

    train_loader = torch.utils.data.DataLoader(AudioDataset())

    val_loader = torch.utils.data.DataLoader(AudioDataset())

    net = VGG16()
    net = net.to(device)
    optimizer = get_optimizer(net=net, wd=wd, lr=lr, momentum=momentum)
    loss = get_loss_function()

    top_accuracy = train(file=file, net=net, train_loader=train_loader, val_loader=val_loader,
                         optimizer=optimizer, cost_function=loss, n_classes=n_classes, n_clips=n_clips,
                         batch_size=batch_size)

    file.close()
    print('TOP TEST ACCURACY:' + str(top_accuracy))
    print('THE END!!!')


if __name__ == '__main__':
    main()
