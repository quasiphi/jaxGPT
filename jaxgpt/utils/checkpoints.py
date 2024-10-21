import pickle

def save_checkpoint(state, epoch, filename=None):
    fn = filename if filename else f'checkpoint_{epoch}.pkl'
    with open(fn, 'wb') as f:
        pickle.dump(state.params, f)


def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)