import matplotlib.pyplot as plt

def plot_lr_accuracy(sgd_accuracy, lion_accuracy, sophia_accuracy, adam_accuracy, lrs, ylabel, filename):
    """
    Plot accuracy/loss vs learning rate
    @param sgd_accuracy: list of accuracies/losses for SGD (float[])
    @param lion_accuracy: list of accuracies/losses for Lion (float[])
    @param sophia_accuracy: list of accuracies/losses for Sophia (float[])
    @param adam_accuracy: list of accuracies/losses for Adam (float[])
    @param lrs: list of learning rates (float[])
    @param ylabel: label for y-axis (str)
    @param filename: filename to save plot to (str)
    """
    plt.figure()
    # plot accuracy vs lr
    lrs = list(map(lambda x: str(x), lrs))

    plt.xlabel('Learning Rate')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(lrs, sgd_accuracy, label='SGD')
    plt.plot(lrs, lion_accuracy, label='Lion')
    plt.plot(lrs, sophia_accuracy, label='Sophia')
    plt.plot(lrs, adam_accuracy, label='Adam')

    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=500)



def plot_bs_accuracy(sgd_accuracy, lion_accuracy, sophia_accuracy, adam_accuracy, batch_sizes, ylabel, filename):
    """
    Plot accuracy/loss vs batch size
    @param sgd_accuracy: list of accuracies/losses for SGD (float[])
    @param lion_accuracy: list of accuracies/losses for Lion (float[])
    @param sophia_accuracy: list of accuracies/losses for Sophia (float[])
    @param adam_accuracy: list of accuracies/losses for Adam (float[])
    @param batch_sizes: list of batch sizes (int[])
    @param ylabel: label for y-axis (str)
    @param filename: filename to save plot to (str)
    """
    plt.figure()
    # plot accuracy vs lr
    batch_sizes = list(map(lambda x: str(x), batch_sizes))

    plt.xlabel('Batch Size')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(batch_sizes, sgd_accuracy, label='SGD')
    plt.plot(batch_sizes, lion_accuracy, label='Lion')
    plt.plot(batch_sizes, sophia_accuracy, label='Sophia')
    plt.plot(batch_sizes, adam_accuracy, label='Adam')

    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=500)
