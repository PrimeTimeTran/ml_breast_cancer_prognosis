import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

from .dcm_reader import view_image
from .utils import base_dir, get_data_file, get_files


class DataPreProcessor():
    @classmethod
    def update_paths(cls, set_type):
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
    def update_column_index(cls, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        cols = list(df.columns)
        cols[1], cols[2] = cols[2], cols[1]
        df = df[cols]
        df.to_csv(csv_file, index=False)

    @classmethod
    def sort_by_patient_ids(cls, patient_id):
        match = re.match(r"([a-zA-Z]+)([0-9]+)", patient_id)
        if match:
            return match.group(1), int(match.group(2))
        return patient_id, 0

    @classmethod
    def update_folder_name(cls, row, set_type):
        search_dir = os.path.join(
            base_dir, f'../tmp/{set_type}/manifest-1617905855234/Breast-Cancer-Screening-DBT')
        if len(row) > 2 and row[3] != None:
            old_folder_path = row[3]
            old_folder_name = old_folder_path.split('/')[-2]
            search_value = old_folder_name.split('-')[0]
            search_path = os.path.join(
                search_dir, row[0], row[3].split('/')[2])
            if os.path.exists(search_path):
                for folder in os.listdir(search_path):
                    if folder.startswith(search_value):
                        new_folder_name = folder
                        new_folder_path = old_folder_path.replace(
                            old_folder_name, new_folder_name)
                        row[3] = new_folder_path
                        break

    @classmethod
    def cleanse_data(cls, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        df_cleaned = df.groupby(['PatientID', 'View'], as_index=False).first()
        cleaned_counts = df_cleaned['PatientID'].value_counts()
        print(cleaned_counts)
        df_cleaned.to_csv(csv_file, index=False)

    @classmethod
    def generate_pngs(cls, set_type):
        csv_file = get_data_file(set_type)
        df = pd.read_csv(csv_file)
        patient_counts = df['PatientID'].value_counts()
        patient_ids_with_2_or_more = patient_counts[patient_counts >= 2].index.tolist(
        )
        patient_ids_with_2_or_more.sort(
            key=DataPreProcessor.sort_by_patient_ids)

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
    def sort_rows(cls, set_type, scope):
        for _, file in enumerate(get_files(set_type, scope)):
            df = pd.read_csv(file)
            df = df.sort_values(
                by='PatientID', key=lambda x: x.map(DataPreProcessor.sort_by_patient_ids))
            df.to_csv(file, index=False)
        print('Done sorting')
    
    @classmethod
    def underfit(cls, set_type, scope):
        items = ['DBT-P00003', 'DBT-P00004', 'DBT-P00005', 'DBT-P00006', 'DBT-P00007', 'DBT-P00008', 'DBT-P00009', 'DBT-P00010', 'DBT-P00011', 'DBT-P00012', 'DBT-P00014', 'DBT-P00015', 'DBT-P00016', 'DBT-P00017', 'DBT-P00019', 'DBT-P00020', 'DBT-P00021', 'DBT-P00022', 'DBT-P00025', 'DBT-P00026', 'DBT-P00027', 'DBT-P00028', 'DBT-P00029', 'DBT-P00030', 'DBT-P00031', 'DBT-P00032', 'DBT-P00033', 'DBT-P00034', 'DBT-P00035', 'DBT-P00036', 'DBT-P00037', 'DBT-P00038', 'DBT-P00039', 'DBT-P00040', 'DBT-P00041', 'DBT-P00042', 'DBT-P00043', 'DBT-P00044', 'DBT-P00045', 'DBT-P00046', 'DBT-P00048', 'DBT-P00049', 'DBT-P00050', 'DBT-P00051', 'DBT-P00052', 'DBT-P00053', 'DBT-P00054', 'DBT-P00055', 'DBT-P00056', 'DBT-P00057', 'DBT-P00058', 'DBT-P00059', 'DBT-P00061', 'DBT-P00062', 'DBT-P00063', 'DBT-P00064', 'DBT-P00065', 'DBT-P00066', 'DBT-P00067', 'DBT-P00068', 'DBT-P00070', 'DBT-P00071', 'DBT-P00072', 'DBT-P00073', 'DBT-P00074', 'DBT-P00075', 'DBT-P00076', 'DBT-P00077', 'DBT-P00078', 'DBT-P00079', 'DBT-P00080', 'DBT-P00081', 'DBT-P00083', 'DBT-P00085', 'DBT-P00086', 'DBT-P00088', 'DBT-P00089', 'DBT-P00090', 'DBT-P00091', 'DBT-P00092', 'DBT-P00093', 'DBT-P00094', 'DBT-P00095', 'DBT-P00096', 'DBT-P00097', 'DBT-P00098', 'DBT-P00099', 'DBT-P00100', 'DBT-P00101', 'DBT-P00102', 'DBT-P00103', 'DBT-P00104', 'DBT-P00105', 'DBT-P00106', 'DBT-P00108', 'DBT-P00109', 'DBT-P00110', 'DBT-P00111', 'DBT-P00112', 'DBT-P00113', 'DBT-P00115', 'DBT-P00116', 'DBT-P00117', 'DBT-P00118', 'DBT-P00119', 'DBT-P00120', 'DBT-P00121', 'DBT-P00122', 'DBT-P00123', 'DBT-P00124', 'DBT-P00125', 'DBT-P00126', 'DBT-P00127', 'DBT-P00128', 'DBT-P00129', 'DBT-P00130', 'DBT-P00131', 'DBT-P00132', 'DBT-P00134', 'DBT-P00135', 'DBT-P00136', 'DBT-P00137', 'DBT-P00138', 'DBT-P00139', 'DBT-P00140', 'DBT-P00141', 'DBT-P00142', 'DBT-P00143', 'DBT-P00144', 'DBT-P00145', 'DBT-P00146', 'DBT-P00147', 'DBT-P00148', 'DBT-P00149', 'DBT-P00150', 'DBT-P00151', 'DBT-P00152', 'DBT-P00153', 'DBT-P00154', 'DBT-P00155', 'DBT-P00156', 'DBT-P00157', 'DBT-P00158', 'DBT-P00159', 'DBT-P00162', 'DBT-P00163', 'DBT-P00164', 'DBT-P00166', 'DBT-P00167', 'DBT-P00168', 'DBT-P00169', 'DBT-P00170', 'DBT-P00171', 'DBT-P00172', 'DBT-P00173', 'DBT-P00175', 'DBT-P00177', 'DBT-P00178', 'DBT-P00179', 'DBT-P00180', 'DBT-P00181', 'DBT-P00182', 'DBT-P00184', 'DBT-P00185', 'DBT-P00186', 'DBT-P00188', 'DBT-P00189', 'DBT-P00190', 'DBT-P00191', 'DBT-P00192', 'DBT-P00193', 'DBT-P00195', 'DBT-P00196', 'DBT-P00197', 'DBT-P00198', 'DBT-P00199', 'DBT-P00200', 'DBT-P00201', 'DBT-P00202', 'DBT-P00203', 'DBT-P00204', 'DBT-P00205', 'DBT-P00206', 'DBT-P00207', 'DBT-P00208', 'DBT-P00209', 'DBT-P00210', 'DBT-P00211', 'DBT-P00212', 'DBT-P00214', 'DBT-P00023', 'DBT-P00087', 'DBT-P00161', 'DBT-P00183', 'DBT-P00223', 'DBT-P00259', 'DBT-P00270', 'DBT-P00304', 'DBT-P00310', 'DBT-P00311', 'DBT-P00315', 'DBT-P00339', 'DBT-P00395', 'DBT-P00411', 'DBT-P00455', 'DBT-P00488', 'DBT-P00491', 'DBT-P00499', 'DBT-P00556', 'DBT-P00561', 'DBT-P00642', 'DBT-P00644', 'DBT-P00661', 'DBT-P00688', 'DBT-P00710', 'DBT-P00754', 'DBT-P00785', 'DBT-P00817', 'DBT-P00822', 'DBT-P00833', 'DBT-P00851', 'DBT-P00858', 'DBT-P00869', 'DBT-P00890', 'DBT-P00892', 'DBT-P00910', 'DBT-P00933', 'DBT-P00935', 'DBT-P00942', 'DBT-P00946', 'DBT-P00984', 'DBT-P00986', 'DBT-P00988', 'DBT-P00992', 'DBT-P01010', 'DBT-P01021', 'DBT-P01023', 'DBT-P01027', 'DBT-P01042', 'DBT-P01054', 'DBT-P01068', 'DBT-P01106', 'DBT-P01157', 'DBT-P01184', 'DBT-P01194', 'DBT-P01202', 'DBT-P01211', 'DBT-P01238', 'DBT-P01247', 'DBT-P01292', 'DBT-P01319', 'DBT-P01373', 'DBT-P01388', 'DBT-P01411', 'DBT-P01434', 'DBT-P01447', 'DBT-P01476', 'DBT-P01485', 'DBT-P01491', 'DBT-P01517', 'DBT-P01552', 'DBT-P01553', 'DBT-P01668', 'DBT-P01687', 'DBT-P01725', 'DBT-P01761', 'DBT-P01796', 'DBT-P01822', 'DBT-P01868', 'DBT-P01934', 'DBT-P01940', 'DBT-P01973', 'DBT-P01978', 'DBT-P01991', 'DBT-P01999', 'DBT-P02112', 'DBT-P02118', 'DBT-P02136', 'DBT-P02143', 'DBT-P02161', 'DBT-P02178', 'DBT-P02187', 'DBT-P02191', 'DBT-P02205', 'DBT-P02224', 'DBT-P02236', 'DBT-P02244', 'DBT-P02250', 'DBT-P02281', 'DBT-P02298', 'DBT-P02316', 'DBT-P02365', 'DBT-P02370', 'DBT-P02384', 'DBT-P02390', 'DBT-P02420', 'DBT-P02435', 'DBT-P02447', 'DBT-P02461', 'DBT-P02479', 'DBT-P02502', 'DBT-P02504', 'DBT-P02525', 'DBT-P02544', 'DBT-P02554', 'DBT-P02572', 'DBT-P02600', 'DBT-P02609', 'DBT-P02651', 'DBT-P02680', 'DBT-P02684', 'DBT-P02695', 'DBT-P02725', 'DBT-P02794', 'DBT-P02821', 'DBT-P02826', 'DBT-P02865', 'DBT-P02868', 'DBT-P02885', 'DBT-P02891', 'DBT-P02896', 'DBT-P02901', 'DBT-P02910', 'DBT-P02928', 'DBT-P02941', 'DBT-P02954', 'DBT-P02967', 'DBT-P02977', 'DBT-P03010', 'DBT-P03014', 'DBT-P03051', 'DBT-P03075', 'DBT-P03093', 'DBT-P03120', 'DBT-P03150', 'DBT-P03170', 'DBT-P03172', 'DBT-P03225', 'DBT-P03231', 'DBT-P03273', 'DBT-P03307', 'DBT-P03332', 'DBT-P03357', 'DBT-P03376', 'DBT-P03413', 'DBT-P03417', 'DBT-P03434', 'DBT-P03462', 'DBT-P03479', 'DBT-P03514', 'DBT-P03519', 'DBT-P03537', 'DBT-P03568', 'DBT-P03636', 'DBT-P03637', 'DBT-P03680', 'DBT-P03685', 'DBT-P03695', 'DBT-P03716', 'DBT-P03733', 'DBT-P03749', 'DBT-P03842', 'DBT-P03844', 'DBT-P03845', 'DBT-P03855', 'DBT-P03877', 'DBT-P03964', 'DBT-P03971', 'DBT-P04378', 'DBT-P04391', 'DBT-P04439', 'DBT-P04451', 'DBT-P04588', 'DBT-P04619', 'DBT-P04788', 'DBT-P04802', 'DBT-P04828', 'DBT-P04844', 'DBT-P05049', 'DBT-P00013', 'DBT-P00060', 'DBT-P00221', 'DBT-P00225', 'DBT-P00332', 'DBT-P00361', 'DBT-P00472', 'DBT-P00629', 'DBT-P00684', 'DBT-P00715', 'DBT-P00784', 'DBT-P00794', 'DBT-P00818', 'DBT-P00827', 'DBT-P01130', 'DBT-P01181', 'DBT-P01241', 'DBT-P01262', 'DBT-P01282', 'DBT-P01439', 'DBT-P01461', 'DBT-P01488', 'DBT-P01497', 'DBT-P01563', 'DBT-P01587', 'DBT-P01626', 'DBT-P01670', 'DBT-P01702', 'DBT-P01718', 'DBT-P01751', 'DBT-P01753', 'DBT-P01817', 'DBT-P01837', 'DBT-P01839', 'DBT-P01898', 'DBT-P02009', 'DBT-P02065', 'DBT-P02139', 'DBT-P02152', 'DBT-P02171', 'DBT-P02227', 'DBT-P02380', 'DBT-P02471', 'DBT-P02493', 'DBT-P02511', 'DBT-P02579', 'DBT-P02588', 'DBT-P02685', 'DBT-P02736', 'DBT-P02750', 'DBT-P02798', 'DBT-P02843', 'DBT-P02919', 'DBT-P03009', 'DBT-P03017', 'DBT-P03064', 'DBT-P03073', 'DBT-P03085', 'DBT-P03176', 'DBT-P03203', 'DBT-P03212', 'DBT-P03218', 'DBT-P03423', 'DBT-P03430', 'DBT-P03458', 'DBT-P03539', 'DBT-P03658', 'DBT-P03677', 'DBT-P03732', 'DBT-P03748', 'DBT-P03812', 'DBT-P03816', 'DBT-P03864', 'DBT-P04255', 'DBT-P04479', 'DBT-P04499', 'DBT-P04859', 'DBT-P04910', 'DBT-P04935', 'DBT-P04975', 'DBT-P05036', 'DBT-P00107', 'DBT-P00194', 'DBT-P00277', 'DBT-P00303', 'DBT-P00318', 'DBT-P00387', 'DBT-P00538', 'DBT-P00583', 'DBT-P00654', 'DBT-P00659', 'DBT-P00675', 'DBT-P00770', 'DBT-P00798', 'DBT-P00801', 'DBT-P00838', 'DBT-P00882', 'DBT-P01110', 'DBT-P01112', 'DBT-P01139', 'DBT-P01158', 'DBT-P01183', 'DBT-P01210', 'DBT-P01267', 'DBT-P01347', 'DBT-P01367', 'DBT-P01493', 'DBT-P01539', 'DBT-P01624', 'DBT-P01673', 'DBT-P01712', 'DBT-P01745', 'DBT-P01801', 'DBT-P01803', 'DBT-P01826', 'DBT-P01958', 'DBT-P02100', 'DBT-P02133', 'DBT-P02164', 'DBT-P02172', 'DBT-P02176', 'DBT-P02308', 'DBT-P02347', 'DBT-P02510', 'DBT-P02532', 'DBT-P02582', 'DBT-P02714', 'DBT-P02738', 'DBT-P02774', 'DBT-P02834', 'DBT-P02935', 'DBT-P03220', 'DBT-P03222', 'DBT-P03292', 'DBT-P03398', 'DBT-P03571', 'DBT-P04413', 'DBT-P04571', 'DBT-P04956', 'DBT-P04982']
        for _, file in enumerate(get_files(set_type, scope)):
            df = pd.read_csv(file)
            df_filtered = df[df['PatientID'].isin(items)]
            df_filtered.to_csv(file, index=False)
        print('Done under fitting')

    @classmethod
    def check_collisions(cls, dir1, dir2):
        collisions = []
        filenames = {}

        for directory in [dir1, dir2]:
            for root, _, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    basename, _ = os.path.splitext(filename)
                    if basename in filenames:
                        collisions.append((filenames[basename], filepath))
                    else:
                        filenames[basename] = filepath

            return collisions