import pickle

def save_file(content, filename):
    with open(f'{filename}.pkl', "wb") as output_file:
        pickle.dump(content, output_file)

def load_file(filename):
    with open(f'{filename}.pkl', "rb") as output_file:
        content = pickle.load(output_file)
    return content