import tensorflow as tf

class FM(tf.keras.layers.Layer):
    """显示特征交叉，直接按照优化后的公式实现即可
    注意：
        1. 传入进来的参数看起来是一个Embedding权重，没有像公式中出现的特征，那是因
        为，输入的id特征本质上都是onehot编码，取出对应的embedding就等价于特征乘以
        权重。所以后续的操作直接就是对特征进行操作
        2. 在实现过程中，对于公式中的平方的和与和的平方两部分，需要留意是在哪个维度
        上计算，这样就可以轻松实现FM特征交叉模块
    """
    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('`FM` layer should be called \
                on a list of at least 2 inputs')
        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        """
        inputs: 是一个列表，列表中每个元素的维度为：(None, 1, emb_dim)， 列表长度
            为field_num
        """
        inputs = [tf.expand_dims(input, 1) for input in inputs]
        concated_embeds_value =  tf.concat(inputs, axis=1) #(None,field_num,emb_dim)
        # 根据最终优化的公式计算即可，需要注意的是计算过程中是沿着哪个维度计算的，将代码和公式结合起来看会更清晰
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keepdims=True)) # (None, 1, emb_dim)
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value,
             axis=1, keepdims=True) # (None, 1, emb_dim)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)#(None,1)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
    
    def get_config(self):
        return super().get_config()

def deepfm_keras_preprocess(input_list, fields, actvation):
    for i in range(len(fields)):
        if fields[i].shape[1]==1:
            fields[i] = tf.squeeze(fields[i], 1)
    fm = FM()
    output = fm(fields)
    if actvation is None:
        pass
    elif actvation == "relu":
        output = tf.keras.layers.ReLU()(output)
    elif actvation == "sigmoid":
        output = tf.keras.layers.Activation("sigmoid")(output)
    model = tf.keras.Model(input_list, output)
    return model