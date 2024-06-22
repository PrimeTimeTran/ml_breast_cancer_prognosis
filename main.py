import sys
from src.sk_learn_wrapper import save_tensor

from src.model import Model
from src.data_preprocessor import DataPreProcessor

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <dataset_name> <fn_name>")
        sys.exit(1)

    [_, dataset_name, fn_name] = sys.argv

    if fn_name == 'update_paths':
        DataPreProcessor.update_paths(dataset_name)
    elif fn_name == 'cleanse_data':
        DataPreProcessor.cleanse_data(dataset_name)
    elif fn_name == 'generate_pngs':
        DataPreProcessor. generate_pngs(dataset_name)
    elif fn_name == 'update_column_index':
        DataPreProcessor.update_column_index(dataset_name)
    elif fn_name == 'sort_rows':
        DataPreProcessor.sort_rows(dataset_name)
    elif fn_name == 'train_model':
        Model('KNN')
    elif fn_name == 'save_tensor':
        save_tensor()
    else:
        print(f"Function {fn_name} is not recognized.")
        sys.exit(1)
