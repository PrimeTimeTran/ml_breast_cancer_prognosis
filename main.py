import sys
import time
import argparse
from src.model import Model
from src.data_preprocessor import DataPreProcessor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ds", "--data_set", type=str, default='train')
    ap.add_argument("-ts", "--train_scope", type=str, default='part')
    ap.add_argument("-m", "--method", type=str, default='train_model')
    ap.add_argument("-mt", "--model_type", type=str, default='KNN')
    args = vars(ap.parse_args())

    fn_name = args['method']
    dataset_name = args['data_set']
    train_scope = args['train_scope']
    model_type = args['model_type']

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
        start_time = time.time()
        model = Model('KNN', train_scope)
        model.classifier = model.select_classifier()
        model.train()
        model.log_train_summary(start_time)
    elif fn_name == 'check_collisions':
        data_dir1 = "/Users/future/Documents/Work/_Main/.Projects/ML_DBT_Classifier/tmp/train/manifest-1617905855234/Breast-Cancer-Screening-DBT"
        data_dir2 = "/Users/future/Documents/Work/_Main/.Projects/ML_DBT_Classifier/tmp/test/manifest-1617905855234/Breast-Cancer-Screening-DBT"
        collision_pairs = DataPreProcessor.check_collisions(data_dir1, data_dir2)

        if collision_pairs:
            print("Collisions found:")
        for pair in collision_pairs:
            print(f"File 1: {pair[0]}")
            print(f"File 2: {pair[1]}\n")
        else:
            print("No collisions found between directories.")

    elif fn_name == 'plot_points':
        model = Model.from_pickle(f'tmp/models/{model_type.lower()}_{train_scope.lower()}_classifier.pickle')
        model.classifier = model.select_classifier()
        model.fit_classifier()
        model.render_knn_plot()
    else:
        print(f"Function {fn_name} is not recognized.")
        sys.exit(1)
