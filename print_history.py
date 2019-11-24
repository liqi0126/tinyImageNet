# -*- coding: utf-8 -*-
file_name = 'history-resnext-101'
import matplotlib.pyplot as plt


# epoch, train_acc, train_loss, test_acc, test_loss
history = [[], [], [], [], []]
with open(file_name, 'r') as f:
    for line in f:
        line = line.split('\t')
        
        epoch = int(line[0])
        train_acc = float(line[1][7:14])
        train_loss = float(line[2])
        val_acc = float(line[3][7:14])
        val_loss = float(line[4])
        
        history[0].append(epoch)
        history[1].append(train_acc)
        history[2].append(train_loss)
        history[3].append(val_acc)
        history[4].append(val_loss)


plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.plot(history[0], history[1], 'b', label = 'train')
plt.plot(history[0], history[3], 'r', label = 'val')
plt.title('Accuracy-epochs')
plt.legend()
plt.show()

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.plot(history[0], history[2], 'b', label = 'train')
plt.plot(history[0], history[4], 'r', label = 'val')
plt.title('Loss-epochs')
plt.legend()
plt.show()