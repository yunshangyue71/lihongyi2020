from dataset import *
from tools import  _shuffle, _gradient, _forward, _accuracy, _cross_entropy_loss

dp = DataProcess()
train, val, test = dp.run()

train_x = train[:, :-1]
train_y = train[:,-1]
val_x = val[:, :-1]
val_y = val[:, -1]
test_x = test[:, :-1]
test_y = test[:, -1]

epoches = 20000
batch_size = 256
learning_rate = 0.01
step = 1
load_model_flag = 1
model_name = "e"
test_flag = 1

train_size, feature_num = train_x.shape
val_size, _ = val_x.shape
test_size, _ = test_x.shape

w = np.zeros((feature_num,))
b = np.zeros((1,))

train_losss = []
train_accs = []
val_losss = []
val_accs = []
if load_model_flag:
    w = np.load(root + "/ckpt/w_"+model_name+".npy")
    b = np.load(root + "/ckpt/b_"+model_name+".npy")

val_best_acc = 0.0
# Iterative training
for epoch in range(epoches):
    if test_flag:
        break
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(train_x, train_y)
    train_loss = []
    train_acc = []
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1

        train_y_pred = _forward(train_x, w, b)
        train_a = _accuracy(train_y_pred, train_y)
        train_l = _cross_entropy_loss(train_y_pred, train_y) / train_y_pred.shape[0]
        l2_l = np.sqrt(np.sum(w**2)) * 1e-4
        train_l += l2_l
        train_acc.append(train_a)
        train_loss.append(train_l)



    if epoch % 20 == 0 and epoch > 1:
        train_accs = np.mean(train_acc)
        train_losss = np.mean(train_loss)

        val_y_pred = _forward(val_x, w, b)
        # test_y_pred = np.round(test_y_pred)
        val_acc = _accuracy(val_y_pred, val_y)
        val_loss = _cross_entropy_loss(val_y_pred, val_y) / val_size
        val_losss.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > val_best_acc:
            np.save(root + "/ckpt/w_" + model_name, w)
            np.save(root + "/ckpt/b_" + model_name, b)
        print("epoch: %d, train_loss: %2.4f, train_acc: %2.4f, val_loss: %2.4f, val_acc: %2.4f, lr: %1.6f  "%(epoch, np.mean(train_loss), np.mean(train_acc), val_loss, val_acc, learning_rate))



test_y_pred = _forward(test_x, w, b)
test_acc  = _accuracy(test_y_pred, test_y)
test_loss  = _cross_entropy_loss(test_y_pred, test_y) /test_size
print(test_acc, test_loss)

print()