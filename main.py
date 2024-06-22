import sys
import time
import argparse
from src.model import Model
from src.data_preprocessor import DataPreProcessor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--method", type=str, default='train_model')
    ap.add_argument("-ds", "--data_set", type=str, default='train')
    ap.add_argument("-cs", "--classification_strategy", type=str, default='KNN')
    ap.add_argument("-ts", "--train_scope", type=str, default='part')
    args = vars(ap.parse_args())

    fn_name = args['method']
    dataset_name = args['data_set']
    train_scope = args['train_scope']
    classification_strategy = args['classification_strategy']

    if fn_name == 'update_paths':
        DataPreProcessor.update_paths(dataset_name)
    if fn_name == 'underfit':
        DataPreProcessor.underfit(dataset_name, train_scope)
    elif fn_name == 'cleanse_data':
        DataPreProcessor.cleanse_data(dataset_name)
    elif fn_name == 'generate_pngs':
        DataPreProcessor.generate_pngs(dataset_name)
    elif fn_name == 'update_column_index':
        DataPreProcessor.update_column_index(dataset_name)
    elif fn_name == 'sort_rows':
        DataPreProcessor.sort_rows(dataset_name, train_scope)
    elif fn_name == 'train_model':
        start_time = time.time()
        model = Model(classification_strategy, train_scope, dataset_name)
        model.classifier = model.select_classifier()
        model.train()
        model.log_train_summary(start_time)
        model.render_sampled_test_imgs_with_labels()
        model.render_knn_plot()
        model.render_matrix('test')
    elif fn_name == 'plot_points':
        model = Model.from_pickle(f'tmp/models/{classification_strategy.lower()}_{train_scope.lower()}_classifier.pickle')
        model.classifier = model.select_classifier()
        model.fit_classifier()
        model.render_knn_plot()
    else:
        print(f"Function {fn_name} is not recognized.")
        sys.exit(1)
