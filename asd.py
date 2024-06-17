import matplotlib.pyplot as plt

# Data from the first table
sample_sizes = [130, 260, 450, 702]
svm_train_times = [0.12, 0.14, 0.17, 0.2]
nn_train_times = [2.5, 3.98, 5.12, 10.23]

# Data from the second table
accuracies = [50, 52, 73, 95,]

# Create the first plot for the training times
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, svm_train_times, label='Время обучения SVM', linestyle='--')
plt.plot(sample_sizes, nn_train_times, label='Время обучения нейросети')
plt.xlabel('Размер множества')
plt.ylabel('Время обучения (сек.)')
plt.title('Время обучения SVM и нейросети в зависимости от размера обучающего множества')
plt.legend()
plt.grid()
plt.savefig('training_times.pdf')

# Create the second plot for the accuracies
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, accuracies, label='Точность')
plt.xlabel('Размер множества')
plt.ylabel('Точность (%)')
plt.title('Точность в зависимости от размера обучающего множества')
plt.legend()
plt.grid()
plt.savefig('accuracies.pdf')

print('Graphs saved as "training_times.png" and "accuracies.png"')