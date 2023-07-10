import os
import shutil
import pandas as pd


def prepare_data(train_list, test_list, data_pd, targetnames, train_dir, test_dir):
    # Set the image_id as the index in data_pd
    data_pd.set_index('image_id', inplace=True)

    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for i in targetnames:
        directory1 = os.path.join(train_dir, i)
        directory2 = os.path.join(test_dir, i)
        os.mkdir(directory1)
        os.mkdir(directory2)

    for image in train_list:
        file_name = image + '.jpg'
        label = data_pd.loc[image, 'dx']
        source = os.path.join('data', 'HAM10000', file_name)
        target = os.path.join(train_dir, label, file_name)
        shutil.copyfile(source, target)

    for image in test_list:
        file_name = image + '.jpg'
        label = data_pd.loc[image, 'dx']
        source = os.path.join('data', 'HAM10000', file_name)
        target = os.path.join(test_dir, label, file_name)
        shutil.copyfile(source, target)


def create_train_test_split(data_pd, test_df, train_dir):
    def identify_train_or_test(x):
        test_data = set(test_df['image_id'])
        if str(x) in test_data:
            return 'test'
        else:
            return 'train'

    data_pd['train_test_split'] = data_pd['image_id'].apply(identify_train_or_test)
    train_df = data_pd[data_pd['train_test_split'] == 'train']
    lesion_id_count = train_df.groupby('lesion_id').count()
    lesion_id_count = lesion_id_count[lesion_id_count['dx'] == 1]
    lesion_id_count.reset_index(inplace=True)

    def duplicates(x):
        unique = set(lesion_id_count['lesion_id'])
        if x in unique:
            return 'no'
        else:
            return 'duplicates'

    data_pd['is_duplicate'] = data_pd['lesion_id'].apply(duplicates)

    df_count = data_pd[data_pd['is_duplicate'] == 'no']
    train, test_df = train_test_split(df_count, test_size=0.15, stratify=df_count['dx'])

    return train_df, test_df
