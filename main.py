import sys
from src.data_prep import update_paths, cleanse_data, generate_pngs, sort_rows

from src.model import Model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/data_prep.py <set_type> <fn_name>")
        sys.exit(1)

    set_type = sys.argv[1]
    fn_name = sys.argv[2]

    if fn_name == 'update_paths':
        update_paths(set_type)
    elif fn_name == 'cleanse_data':
        cleanse_data(set_type)
    elif fn_name == 'generate_pngs':
        generate_pngs(set_type)
    elif fn_name == 'sort_rows':
        sort_rows(set_type)
    elif fn_name == 'train_model':
        Model('KNN')
    else:
        print(f"Function {fn_name} is not recognized.")
        sys.exit(1)
