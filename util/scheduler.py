class prune_scheduler:
    def __init__(self, total_iters, epochs):
        self.total_iters = total_iters
        self.epochs = epochs

        interval = epochs / total_iters
        self.prune_epochs = [round(i * interval) for i in range(total_iters)]
        self.prune_epochs = sorted(set(self.prune_epochs))

    def __call__(self, epoch):
        return epoch in self.prune_epochs