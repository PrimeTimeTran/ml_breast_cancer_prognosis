import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

from .dcm_reader import view_image
from .utils import base_dir, get_data_file, get_files

class DataPreProcessor():
    @classmethod
    def update_column_index(self, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        cols = list(df.columns)
        cols[1], cols[2] = cols[2], cols[1]
        df = df[cols]
        df.to_csv(csv_file, index=False)

    @classmethod
    def sort_by_patient_ids(self, patient_id):
        match = re.match(r"([a-zA-Z]+)([0-9]+)", patient_id)
        if match:
            return match.group(1), int(match.group(2))
        return patient_id, 0

    @classmethod
    def update_folder_name(self, row, set_type):
        search_dir = os.path.join(
            base_dir, f'../tmp/{set_type}/manifest-1617905855234/Breast-Cancer-Screening-DBT')
        if len(row) > 2 and row[3] != None:
            old_folder_path = row[3]
            old_folder_name = old_folder_path.split('/')[-2]
            search_value = old_folder_name.split('-')[0]
            search_path = os.path.join(search_dir, row[0], row[3].split('/')[2])
            if os.path.exists(search_path):
                for folder in os.listdir(search_path):
                    if folder.startswith(search_value):
                        new_folder_name = folder
                        new_folder_path = old_folder_path.replace(
                            old_folder_name, new_folder_name)
                        row[3] = new_folder_path
                        break
    @classmethod
    def update_paths(self, set_type):
        csv_file = get_data_file(set_type)
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        header = rows[0]
        data_rows = rows[1:]

        for row in data_rows:
            DataPreProcessor.update_folder_name(row, set_type)

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data_rows)

        print("CSV file has been updated successfully.")

    @classmethod
    def cleanse_data(self, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        df_cleaned = df.groupby(['PatientID', 'View'], as_index=False).first()
        cleaned_counts = df_cleaned['PatientID'].value_counts()
        print(cleaned_counts)
        df_cleaned.to_csv(csv_file, index=False)
    
    @classmethod
    def generate_pngs(self, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        patient_counts = df['PatientID'].value_counts()
        patient_ids_with_2_or_more = patient_counts[patient_counts >= 2].index.tolist()
        patient_ids_with_2_or_more.sort(key=DataPreProcessor.sort_by_patient_ids)

        print(f'patient_counts {patient_counts}')
        for _, patient_id in enumerate(patient_ids_with_2_or_more):
            # INFO: Used to "pick up where you left off" in generating images
            # num = int(re.findall(r'\d+', patient_id)[0])
            # if num < 3906:
            #     continue

            patient_rows = df[df['PatientID'] == patient_id]
            for _, row in patient_rows.iterrows():
                try:
                    specific_row = df[(df.PatientID == row[0])
                                    & (df.View == row[1])]
                    if not specific_row.empty:
                        actual_row_number = specific_row.index[0]
                        view_image(set_type, df, plt, actual_row_number)
                    else:
                        print(f'specific_row {specific_row}')
                except Exception as e:
                    print(f"An error occurred: {e}")

    @classmethod
    def sort_rows(self, set_type):
        for _, file in enumerate(get_files(set_type)):
            df = pd.read_csv(file)
            df = df.sort_values(
                by='PatientID', key=lambda x: x.map(DataPreProcessor.sort_by_patient_ids))
            df.to_csv(file, index=False)
