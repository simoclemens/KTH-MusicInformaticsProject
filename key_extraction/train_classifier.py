from utils.train_functions import get_loss_function, get_optimizer, compute_accuracy, getTop5Correct
import torch.nn.parallel
import torch.optim
import torch
from utils.data_loader import AudioTrackDataset
import os
import torch.utils.data
from models.classifier import KeyClassifier
import datetime


# train function
def train(file, net, train_loader, val_loader, optimizer, cost_function, n_classes, batch_size=32,
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
        inputs = data_source['features'].view(32, 200).to(torch.float32).to(device)

        logits = net.forward(inputs)  # get predictions from the net

        loss = cost_function(logits, label)
        loss.backward()  # apply the backward

        optimizer.step()  # update the parameters
        optimizer.zero_grad()  # reset gradient of the optimizer for next iteration

        test_metrics = validate(net, val_loader, n_classes, batch_size)

        file.write('[{}/{}] ITERATION COMPLETED\n'.format(iteration, training_iterations))
        file.write('TEST: acc@top1={:.2f}%  acc@top5={:.2f}%\n\n'.format(test_metrics['top1'] * 100, 0))

        if test_metrics['top1'] > top_accuracy:
            top_accuracy = test_metrics['top1']
            print('ITERATION:' + str(iteration) + ' - BEST ACCURACY: {:.2f}'.format(top_accuracy * 100) +
                  ' - BEST ACCURACY: {:.2f}'.format(test_metrics['top5'] * 100))

            # Save model weights and optimizer state
            torch.save(net.state_dict(), "model_weights.pth")
        if iteration % 10 == 0:
            print('ITERATION:' + str(iteration))

    file.write('TOP ACCURACY {:.2f}'.format(top_accuracy))

    return top_accuracy


# validation function
def validate(net, val_loader, n_classes, batch_size=32, device="cuda:0"):
    net.train(False)  # set model to validate

    total_size = len(val_loader.dataset)
    val_correct = 0
    top5_correct = 0
    counter = {}

    for i in range(n_classes):
        counter[i] = [0, 0]

    with torch.no_grad():  # do not update the gradient
        for iteration, (data_source) in enumerate(val_loader):  # extract batches from the val_loader
            size = data_source['label'].shape[0]
            label = data_source['label'].to(device)  # send label to gpu
            inputs = data_source['features'].view(32, 200).to(torch.float32).to(device)
            logits = net(inputs)  # get predictions from the net

            _, predicted = torch.max(logits.data, 1)
            for i in range(size):
                if predicted[i] == data_source['label'][i].item():
                    counter[data_source['label'][i].item()][0] += 1
                counter[data_source['label'][i].item()][1] += 1

            val_correct += (predicted == label).sum().item()
            top5_correct += getTop5Correct(logits, label)

    # compute the accuracy
    accuracy = val_correct / total_size
    top5_accuracy = top5_correct / total_size
    for i in range(n_classes):
        if counter[i][1] != 0:
            counter[i] = counter[i][0] / counter[i][1]
        else:
            counter[i] = 0
    test_results = {'top1': accuracy, 'top5': top5_accuracy, 'classes_acc': counter}

    return test_results


def main():
    device = "cuda:0"

    lr = 0.02
    wd = 1e-7
    momentum = 0.9
    loss_weight = 1

    batch_size = 32

    log_path = 'logs'

    name = "indices_features"
    n_classes = 24
    annotations_path = "../dataset"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = name + '_' + timestamp + '.txt'

    file = open(os.path.join(log_path, file_name), 'w')

    train_loader = torch.utils.data.DataLoader(AudioTrackDataset(mode='train',
                                                                 annotations_path=annotations_path,
                                                                 name=name,
                                                                 custom_duration=True),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(AudioTrackDataset(mode='test',
                                                               annotations_path=annotations_path,
                                                               name=name,
                                                               custom_duration=True),
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True, drop_last=True)

    net = KeyClassifier()
    net = net.to(device)
    optimizer = get_optimizer(net=net, wd=wd, lr=lr, momentum=momentum)
    loss = get_loss_function()

    top_accuracy = train(file=file, net=net, train_loader=train_loader, val_loader=val_loader,
                         optimizer=optimizer, cost_function=loss, n_classes=n_classes,
                         batch_size=batch_size)

    file.close()
    print('TOP TEST ACCURACY:' + str(top_accuracy))
    print('THE END!!!')


if __name__ == '__main__':
    main()
