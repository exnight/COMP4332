from collections import Counter
from itertools import combinations
import random
from math import sqrt

import numpy as np
import pandas as pd
import json

from keras.layers import Input, Dense, Embedding
from keras.layers import Concatenate, Reshape
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=sess_config))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

random.seed(2019)
np.random.seed(2019)
tf.set_random_seed(2019)

STUDENT_ID = '20420684'

embed_size = 64
deep_blocks = 3
deep_units = 256
fc_blocks = 4
fc_units = 128
lr = 0.00025
epochs = 20
batch_size = 64
patience = int(epochs*0.2)


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


def build_deepwide_model(len_continuous, deep_vocab_lens, len_wide, embed_size):
    input_list = []
    continuous_input = Input(shape=(len_continuous,), dtype='float32', name='Cont_Input')
    input_list.append(continuous_input)

    emb_list = []
    i = 0
    for vocab_size in deep_vocab_lens:
        i += 1
        _input = Input(shape=(1,), dtype='int32', name='Input_'+str(i))
        input_list.append(_input)
        _emb = Embedding(output_dim=embed_size, input_dim=vocab_size, input_length=1, name='Embed_'+str(i))(_input)
        _emb = Reshape((embed_size,), name='Embed_Flat_'+str(i))(_emb)
        emb_list.append(_emb)

    deep_input = Concatenate(name='Concat_Embed_Cont')(emb_list + [continuous_input])
    x = deep_input
    # Create blocks
    # Dense --> Activation --> BatchNorm
    for i in range(deep_blocks):
        x = Dense(deep_units//(2**i), name='Deep_'+str(i+1)+'_Dense')(x)
        x = LeakyReLU(name='Deep_'+str(i+1)+'_Acti')(x)
        x = BatchNormalization(name='Deep_'+str(i+1)+'_BN')(x)

    wide_input = Input(shape=(len_wide,), dtype='float32', name='Wide_Input')
    input_list.append(wide_input)

    # Output Stage
    fc_input = Concatenate(name='Concat_Deep_Wide')([x, wide_input])
    fc = fc_input
    for i in range(fc_blocks):
        fc = Dense(fc_units//(2**i), name='FC_'+str(i+1)+'_Dense')(fc)
        fc = LeakyReLU(name='FC_'+str(i+1)+'_Acti')(fc)
        fc = BatchNormalization(name='FC_'+str(i+1)+'_BN')(fc)

    model_output = Dense(1, name='Output')(fc)
    model = Model(inputs=input_list, outputs=model_output)

    return model


def get_continuous_features(df, continuous_columns):
    continuous_features = df[continuous_columns].values
    return continuous_features


def get_top_k_p_combinations(df, comb_p, topk, output_freq=False):
    def get_category_combinations(categories_str, comb_p=2):
        categories = categories_str.split(', ')

        return list(combinations(categories, comb_p))

    all_categories_p_combos = df["item_categories"].apply(
        lambda x: get_category_combinations(x, comb_p)).values.tolist()
    all_categories_p_combos = [tuple(t) for item in all_categories_p_combos for t in item]

    tmp = dict(Counter(all_categories_p_combos))
    sorted_categories_combinations = list(sorted(tmp.items(), key=lambda x: x[1], reverse=True))

    if output_freq:
        return sorted_categories_combinations[:topk]
    else:
        return [t[0] for t in sorted_categories_combinations[:topk]]


def get_wide_features(df):
    def categories_to_binary_output(categories):
        binary_output = [0 for _ in range(len(selected_categories_to_idx))]
        for category in categories.split(', '):
            if category in selected_categories_to_idx:
                binary_output[selected_categories_to_idx[category]] = 1
            else:
                binary_output[0] = 1

        return binary_output

    def categories_cross_transformation(categories):
        current_category_set = set(categories.split(', '))
        corss_transform_output = [0 for _ in range(len(top_combinations))]
        for k, comb_k in enumerate(top_combinations):
            if len(current_category_set & comb_k) == len(comb_k):
                corss_transform_output[k] = 1
            else:
                corss_transform_output[k] = 0

        return corss_transform_output

    category_binary_features = np.array(df.item_categories.apply(
        lambda x: categories_to_binary_output(x)).values.tolist())
    category_corss_transform_features = np.array(df.item_categories.apply(
        lambda x: categories_cross_transformation(x)).values.tolist())

    return np.concatenate((category_binary_features, category_corss_transform_features), axis=1)


def count_elite(ele):
    if ele == '':
        return 0
    else:
        return len(ele.split(','))


tr_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/valid.csv")
te_df = pd.read_csv("data/test.csv")

tr_ratings = tr_df.stars.values
val_ratings = val_df.stars.values

user_df = pd.read_json("data/user.json")
item_df = pd.read_json("data/business.json")

user_df['elite'] = user_df['elite'].apply(count_elite)

attrs_list = pd.Series(['Alcohol', 'Caters', 'NoiseLevel', 'WiFi'])
attr_cols = attrs_list.str.lower()

for idx in range(item_df.shape[0]):
    try:
        attr = item_df.loc[idx, 'attributes']
    except TypeError:
        attr = "None"

    for i in range(attrs_list.shape[0]):
        attr_name = attrs_list[i]
        attr_col = attr_cols[i]
        if attr != "None":
            try:
                value = attr[attr_name]
            except KeyError:
                value = "Missing"
            except TypeError:
                value = "Missing"

            item_df.at[idx, attr_col] = value
        else:
            item_df.at[idx, attr_col] = attr

user_df = user_df.rename(index=str, columns={t: 'user_' + t for t in user_df.columns if t != 'user_id'})
item_df = item_df.rename(index=str, columns={t: 'item_' + t for t in item_df.columns if t != 'business_id'})

tr_df["index"] = tr_df.index
val_df["index"] = val_df.index
te_df["index"] = te_df.index

tr_df = pd.merge(pd.merge(tr_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)
val_df = pd.merge(pd.merge(val_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)
te_df = pd.merge(pd.merge(te_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)

# Continuous features
print("Prepare continuous features...")
continuous_columns = ["user_average_stars", "user_review_count", "user_useful", "item_is_open", "item_latitude", "item_longitude", "item_review_count", "item_stars"]
tr_continuous_features = get_continuous_features(tr_df, continuous_columns)
val_continuous_features = get_continuous_features(val_df, continuous_columns)
te_continuous_features = get_continuous_features(te_df, continuous_columns)

scaler = StandardScaler().fit(tr_continuous_features)
tr_continuous_features = scaler.transform(tr_continuous_features)
val_continuous_features = scaler.transform(val_continuous_features)
te_continuous_features = scaler.transform(te_continuous_features)

# Deep features
print("Prepare deep features...")
item_deep_columns = ["item_postal_code", "item_alcohol", "item_caters", "item_noiselevel", "item_wifi"]
item_deep_vocab_lens = []
for col_name in item_deep_columns:
    tmp = item_df[col_name].unique()
    vocab = dict(zip(tmp, range(1, len(tmp) + 1)))
    item_deep_vocab_lens.append(len(vocab) + 1)
    item_df[col_name + "_idx"] = item_df[col_name].apply(lambda x: vocab[x] if x in vocab else 0)
item_deep_idx_columns = [t + "_idx" for t in item_deep_columns]
item_to_deep_features = dict(zip(item_df.business_id.values, item_df[item_deep_idx_columns].values.tolist()))

tr_deep_features = np.array(tr_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())
val_deep_features = np.array(val_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())
te_deep_features = np.array(te_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())

# Wide (Category) features
print("Prepare wide features...")

# Prepare binary encoding for each selected categories
all_categories = [category for category_list in item_df.item_categories.values for category in category_list.split(", ")]
category_sorted = sorted(Counter(all_categories).items(), key=lambda x: x[1], reverse=True)
selected_categories = [t[0] for t in category_sorted]
selected_categories_to_idx = dict(zip(selected_categories, range(1, len(selected_categories) + 1)))
selected_categories_to_idx['unk'] = 0
idx_to_selected_categories = {val: key for key, val in selected_categories_to_idx.items()}

# Prepare Cross transformation for each categories
top_combinations = []
top_combinations += get_top_k_p_combinations(tr_df, 2, 16, output_freq=False)
top_combinations += get_top_k_p_combinations(tr_df, 3, 8, output_freq=False)
top_combinations += get_top_k_p_combinations(tr_df, 4, 4, output_freq=False)
top_combinations = [set(t) for t in top_combinations]

tr_wide_features = get_wide_features(tr_df)
val_wide_features = get_wide_features(val_df)
te_wide_features = get_wide_features(te_df)

# Build input
tr_features = []
tr_features.append(tr_continuous_features.tolist())
tr_features += [tr_deep_features[:, i].tolist() for i in range(len(tr_deep_features[0]))]
tr_features.append(tr_wide_features.tolist())

val_features = []
val_features.append(val_continuous_features.tolist())
val_features += [val_deep_features[:, i].tolist() for i in range(len(val_deep_features[0]))]
val_features.append(val_wide_features.tolist())

te_features = []
te_features.append(te_continuous_features.tolist())
te_features += [te_deep_features[:, i].tolist() for i in range(len(te_deep_features[0]))]
te_features.append(te_wide_features.tolist())

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
mc = ModelCheckpoint('build/best_model.h5', save_best_only=True)
cb_list = [es, mc]

# Model training
deepwide_model = build_deepwide_model(len(tr_continuous_features[0]), item_deep_vocab_lens, len(tr_wide_features[0]), embed_size=embed_size)

deepwide_model.compile(optimizer=Adam(lr=lr), loss='mse')

history = deepwide_model.fit(tr_features, tr_ratings, epochs=epochs, batch_size=batch_size, validation_data=(val_features, val_ratings), callbacks=cb_list)

with open('build/log/baseline_history.json', 'w') as fp:
    json.dump(history.history, fp)

# Evaluate model
print(deepwide_model.summary())

best_model = load_model('build/best_model.h5')
y_pred = best_model.predict(tr_features)
print("TRAIN RMSE: ", rmse(y_pred, tr_ratings))
y_pred = best_model.predict(val_features)
print("VALID RMSE: ", rmse(y_pred, val_ratings))

# Make Prediction
y_pred = best_model.predict(te_features)
res_df = pd.DataFrame()
res_df['pred'] = y_pred[:, 0]
res_df.to_csv("build/{}.csv".format(STUDENT_ID), index=False)
print("Writing test predictions to file done.")
