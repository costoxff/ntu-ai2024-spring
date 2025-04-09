import matplotlib.pyplot as plt
import numpy as np

interval = 10

epochs_list = [i * 5 for i in range(interval)]

loss = np.random.randint(0, 20, interval)
reward = np.random.randint(0, 20, interval)

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')

plt.plot(epochs_list, loss, label='loss')
plt.plot(epochs_list, reward, label='reward')
plt.legend()

# plt.savefig("ut1.png")
plt.show()
