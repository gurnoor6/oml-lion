import matplotlib.pyplot as plt

def plot_lr_accuracy(sgd_accuracy, lion_accuracy, sophia_accuracy, adam_accuracy, lrs, ylabel, filename):
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
