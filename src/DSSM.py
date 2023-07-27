import tensorflow as tf

def dssm_model_feature_column(feature_inputs, item_feature_columns, user_feature_columns, item_hidden_unit, user_hidden_units, output_hidden_units, activation = 'sigmoid', use_bn = True):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    output = dssm_model_base(item_tower, user_tower, item_hidden_unit, user_hidden_units, output_hidden_units, activation, use_bn)
    model = tf.keras.Model(feature_inputs, output)
    return model

def dssm_model_keras_preprocess(input_list, item_tower, user_tower, item_hidden_unit, user_hidden_units, output_hidden_units, activation = 'sigmoid', use_bn = True):
    output = dssm_model_base(item_tower, user_tower, item_hidden_unit, user_hidden_units, output_hidden_units, activation, use_bn)
    model = tf.keras.Model(input_list, output)
    return model


def dssm_model_base(item_tower, user_tower, item_hidden_unit, user_hidden_units, output_hidden_units, activation = 'sigmoid', use_bn = True):
    if user_tower.shape[1]==1:
        user_tower = tf.squeeze(user_tower, 1)
    for num_nodes in item_hidden_unit:
        item_tower = tf.keras.layers.Dense(num_nodes, activation= 'relu' )(item_tower)
        if use_bn:
            item_tower = tf.keras.layers.BatchNormalization()(item_tower)
    for  num_nodes  in  user_hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation= 'relu' )(user_tower)
        if use_bn:
            user_tower = tf.keras.layers.BatchNormalization()(user_tower)
    output = tf.keras.layers.Concatenate(axis= -1)([item_tower, user_tower])
    for  num_nodes  in  output_hidden_units:
        output = tf.keras.layers.Dense(num_nodes, activation= 'relu' )(output)
        if use_bn:
            output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense( 1 , activation= activation )(output)
    return  output
