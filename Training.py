import math as M
from progressbar import progressbar


def fit(model, x, y, valsplit=0.2, epochs=1, batchsize=16, loss='mse', momentum=0.9, learning_rate=0.01,
        earlystopping=False, reduceonplateu=False, reducefactor=0.5, patience=3, metrics=[]):
    # put loss as the first metric
    metrics.insert(0, loss)
    # validation split
    splitter = int(len(x)*valsplit)
    x_val = x[:splitter]
    y_val = y[:splitter]
    x_train = x[splitter:]
    y_train = y[splitter:]
    # Train process
    steps = int(len(x_train) / batchsize)
    # Early Stop Parameter
    esp = patience
    history = []
    for i in range(epochs):
        pred = []
        true = []
        print('\nepoch: ' + str(i) + '/' + str(epochs))
        for j in progressbar(range(steps), redirect_stdout=True):
            if (j+1)*batchsize <= len(x):
                for k in range(j*batchsize, (j+1)*batchsize):
                    y_pred = model.feedforward(x_train[k])
                    pred.append(y_pred)
                    true.append(y_train[k])
                    E = Loss(y_pred, y_train[k], loss)
                    model.backpropagate(y_pred, y_train[k], E)
                model.train(learning_rate, momentum)

                # calculate train metrics once every an arbitrary number of batches
                # printing the outstring help us keep track of the model progress through the training process
                if j % int(steps) == 0:
                    train_performance = Metrics(pred, true, metrics)
                    outstring = []
                    for k in range(len(metrics)):
                        outstring.append(metrics[k] + ': ' + str(train_performance[k]) + ', ')
                pred.clear()
                true.clear()
        # calculate metrics on validation data
        for k in progressbar(range(len(x_val)), redirect_stdout=True):
            pred.append(model.feedforward(x_val[k]))
            true.append(y_val[k])
        val_performance = Metrics(pred, true, metrics)
        print('\n variance:'+str(max(pred)[0]-min(pred)[0]))
        outstring = []
        for k in range(len(metrics)):
            outstring.append(metrics[k] + ': ' + str(val_performance[k]) + ', ')

        # Early stopping
        if earlystopping is True and len(history) > 0:
            # pick the smallest error achieved
            h = min(history)
            if val_performance[0] >= h[0]:
                esp -= 1
                # change the learning rate to see if we can get even closer to the loss minimum
                if reduceonplateu is True and esp < (patience/3)+1:
                    learning_rate = learning_rate * reducefactor
                    print('\nReduced')
            else:
                esp = patience
        history.append(val_performance)
        if esp < 1:
            print('\nEarly Stopped')
            break
    print('\nFinished Training')
    return history


def Loss(p, t, loss):
    # the output neurons can be 1, 2 or any number.
    # in case it is only one, we still have to be able to use indexes
    p = [p]
    t = [t]
    l = 0
    if loss == 'mse':
        for k in range(len(p)):
            l += (t[k] - p[k]) ** 2
    elif loss == 'mae':
        for k in range(len(p)):
            l += abs(t[k] - p[k])
    elif loss == 'categorical_crossentropy':
        for k in range(len(p)):
            l += t[k] * M.log(p[k])
    elif loss == 'binary_crossentropy':
        for k in range(len(p)):
            l += -t[k] * M.log(p[k]) - (1 - t[k]) * M.log(1 - p[k])
    l = l / len(p)
    return l



def Metrics(pred, true, metrics):
    performance = []
    t = true
    p = pred
    for i in range(len(metrics)):
        if metrics[i] == 'mse':
            mse = 0
            for j in range(len(p)):
                t[j] = [t[j]]
                for k in range(len(p[j])):
                    mse += (t[j][k] - p[j][k])**2
            performance.append(mse/len(t))
        if metrics[i] == 'mae':
            mae = 0
            for j in range(len(p)):
                t[j] = [t[j]]
                for k in range(len(p[j])):
                    mae += abs(t[j][k] - p[j][k])
            performance.append(mae/len(t))
        if metrics[i] == 'categorical_crossentropy':
            cce = 0
            for j in range(len(p)):
                t[j] = [t[j]]
                for k in range(len(p[j])):
                    cce += t[j][k]*M.log(p[j][k])
            performance.append(-cce/len(t))
        if metrics[i] == 'binary_crossentropy':
            bce = 0
            for j in range(len(p)):
                t[j] = [t[j]]
                for k in range(len(p[j])):
                    bce += -t[j][k]*M.log(p[j][k])-(1-t[j][k])*M.log(1-p[j][k])
            performance.append(bce/len(t))
        if metrics[i] == 'accuracy':
            c = 0
            a = 0
            for j in range(len(p)):
                for k in range(len(p[j])):
                    a += 1
                    # for regression problems, we allow room for error within a
                    # small interval around the target value
                    if abs(p[j][k]-t[j][k]) <= 0.1:
                        c += 1
            performance.append(c/a)
    return performance