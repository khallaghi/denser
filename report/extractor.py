import matplotlib.pyplot as plt
import sys
accuracies = []
best_accuracies = []
if __name__ == '__main__':
    file_input = 'logs/log.txt'
    with open(file_input, 'r+') as f:
        for line in f.readlines():
            if line.find('Accuracy') > -1:
                accuracies.append(float(line[11:]))
    population = 20
    generation = 50
    epochs = 7
    for i in range(generation):
        if i*population > len(accuracies):
            break
        if (i+1)*population < len(accuracies):
            best_accuracies.append(max(accuracies[i*population:(i+1)*population]))
        else:
            best_accuracies.append(max(accuracies[i*population:]))
    x = range(len(best_accuracies))
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, best_accuracies, '-', linewidth=2, markersize=12)
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1])
    # plt.legend(loc='lower right')
    plt.title('The best accuracy in each generation, \npopulation: {}, generation: {}, epochs: {}'.format(population, generation, epochs))
    plt.show()
# print(accuracies)