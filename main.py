from genetic_algorithm.algorithm import ga, fitness, all_losses, all_accuracies

if __name__ == '__main__':
    ga.run()
    fitness(ga.best_individual(), None)
    print("all accuracies: ")
    print(all_accuracies)
    print("all losses: ")
    print(all_losses)
