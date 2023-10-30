'''
layer.py: contains functions used to build all spectral and siamese net models
'''
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Lambda, Dropout
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.regularizers import l2

from . import costs

def orthonorm_op(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x
    正交化输入矩阵x
    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = K.dot(K.transpose(x), x)
    x_2 += K.eye(K.int_shape(x)[1])*epsilon
    L = tf.cholesky(x_2)
    ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
    return ortho_weights

# FFT_layer
def fft_layer(x, output_num, name = None):
    '''
    创建一个keras层，傅里叶变换层

    x:          一个n*d的复数矩阵
    '''

    # 首先根据输入数据计算权重矩阵
    # 1. 获取x的维度
    n = x.get_shape().as_list()[0]
    d = x.get_shape().as_list()[1]

    # 1. 将x转换到复数域
    # x = tf.signal.rfft2d(x)

    # 2. 计算x的相似度矩阵S
    # 2.1 计算x的距离矩阵
    distance = costs.squared_distance(x)
    # 2.2 对距离矩阵进行遍历，若距离小于Derlt，则计算其相似度矩阵中的值，若不负责则相似度矩阵中该位置为0
    # 2.2.1 获取距离矩阵的维度
    distance_row = distance.get_shape().as_list()[0]
    distance_col = distance.get_shape().as_list()[1]
    # 创建存储相似度矩阵S的变量
    S = K.variable(np.zeros((distance_row,distance_col)))
    for i in range(distance_row):
        for j in range(distance_col):
            if distance[i][j]<500:
                S[i][j] = tf.math.exp(-distance/300)
            else:
                S[i][j] = 0
    # 3. 计算x的度矩阵D
    Dm = tf.reduce_sum(S, 1)
    D = Dm
    for k in range(distance_col-1):
        D = tf.concat([D,Dm], axis = 1)
    # 4. 计算x的拉普拉斯矩阵L
    L = tf.matrix_diag(Dm) - S
    # 5. 计算x的特征值
    # 5.1 将x转换到复数域
    x = tf.signal.rfft2d(x)
    # 5.2 将L转换到复数域
    L = tf.signal.rfft2d(L)
    # 5.3 将D转换到复数域
    D = tf.signal.rfft2d(D)
    # 5.4 根据公式计算特征值
    leftItem = tf.reduce_sum(x*x*D, axis = 0)   # 竖向压扁
    rightItem = K.variable(np.zeros((1,distance_col)))
    for p in range(distance_row):
        for q in range(distance_row):
            rightItem = rightItem + x[p]*x[q]*L[p][q]
    myLambda = leftItem*rightItem
    # 6. 根据特征值选取特征向量
    # 6.1 将myLambda转换到实数域 （实部虚部的平方和）
    myLambda_real = K.variable(np.zeros((1, distance_col)))
    for c in range(distance_col):
        c_real = tf.real(myLambda[c])
        c_imag = tf.imag(myLambda[c])
        myLambda_real[c] = tf.add(tf.multiply(c_real, c_real), tf.multiply(c_imag, c_imag))
    # 6.2 对特征值进行从小到大排序，并获取从小到大的索引
    index = np.argsort(myLambda_real)

    # 7. 根据特征值选取特征向量
    # 7.1 定义一个相应维度的傅里叶矩阵（维度应为批次大小）
    F = tf.signal.rfft2d(np.eye(distance_row))
    # 7.2 获取前k（output_num）小特征至对应的特征向量所组成的特征矩阵
    # 7.2.1 创建存储特征矩阵V的变量
    v = np.zeros(distance_row, output_num)
    for i in range(output_num):
        v[:, i] = F[:,i]

    V = K.variable(np.zeros((distance_row, output_num)))
    V = v
    # 8. 根据x计算权重矩阵，权重矩阵为x的转置乘V
    # 8.1 先计算x的转置
    x_z = tf.transpose(x)
    # 8.2 将x的转置转换到复数域
    x_z = tf.signal.rfft2d(x_z)
    # 8.3 x的转置乘V
    w = tf.matmul(x_z, V)
    # 8.4 将权重矩阵转换到实数域（实部虚部的平方和）
    W = np.zeros(d, output_num)
    for c in range(d):
        for d in range(output_num):
            w_real = tf.real(w[c][d])
            w_imag = tf.imag(w[c][d])
            W[c][d] = tf.add(tf.multiply(w_real,w_real), tf.multiply(w_imag, w_imag))
    Weight = K.variable(np.zeros((distance_row, output_num)))
    # 创建将矩阵保存到变量中的操作
    fft_weights_update = tf.assign(Weight, W, name='fft_weights_update')
    # 根据训练或验证在存储权重和计算权重之间切换
    l = Lambda(lambda x: K.in_train_phase(K.dot(x, W), K.dot(x, Weight)))

    l.add_update(fft_weights_update)
    return l



# 正交层的激活函数
def Orthonorm(x, name=None):
    '''
    Builds keras layer that handles orthogonalization of x
    创建一个keras层，用于正交化x

    x:      an n x d input matrix    一个n*d的输入矩阵
    name:   name of the keras layer       keras层的名字

    returns:    a keras layer instance. during evaluation, the instance returns an n x d orthogonal matrix
                if x is full rank and not singular
                一个keras层的实例，在求值期间，如果 x 是全秩而不是单数，则实例返回 N x D 正交矩阵
    '''
    # get dimensionality of x   获取x的维度
    d = x.get_shape().as_list()[-1]
    # compute orthogonalizing matrix
    # 计算x的正交矩阵
    ortho_weights = orthonorm_op(x)
    # create variable that holds this matrix
    # 创建一个变量，存储这个矩阵 d*d
    ortho_weights_store = K.variable(np.zeros((d,d)))
    # create op that saves matrix into variable
    # 创建将矩阵保存到变量中的操作
    ortho_weights_update = tf.assign(ortho_weights_store, ortho_weights, name='ortho_weights_update')
    # switch between stored and calculated weights based on training or validation
    # 根据训练或验证在存储权重和计算权重之间切换
    l = Lambda(lambda x: K.in_train_phase(K.dot(x, ortho_weights), K.dot(x, ortho_weights_store)), name=name)

    l.add_update(ortho_weights_update)
    return l

def stack_layers(inputs, layers, kernel_initializer='glorot_uniform'):
    '''
    Builds the architecture of the network by applying each layer specified in layers to inputs.
    通过将图层中指定的每个层应用于输入来构建网络的体系结构。

    inputs:     a dict containing input_types and input_placeholders for each key and value pair, respecively.
                for spectralnet, this means the input_types 'Unlabeled' and 'Orthonorm'*
                一个字典，包含每个键和值对的input_types和input_placeholders。
                对于Spectralnet，这意味着input_types“未标记”和“正统规范”*

    layers:     a list of dicts containing all layers to be used in the network, where each dict describes
                one such layer. each dict requires the key 'type'. all other keys are dependent on the layer
                type
                包含要在网络中使用的所有层的字典列表，其中每个字典描述一个这样的层。
                每个字典都需要键“type”。所有其他键都取决于图层类型

    kernel_initializer: initialization configuration passed to keras (see keras initializers)
                传递给 Keras 的初始化配置（请参阅 Keras 初始值设定项）

    returns:    outputs, a dict formatted in much the same way as inputs. it contains input_types and
                output_tensors for each key and value pair, respectively, where output_tensors are
                the outputs of the input_placeholders in inputs after each layer in layers is applied
                输出，以与输入大致相同的方式格式化的字典。它分别包含每个键和值对的input_types和output_tensors，
                其中output_tensors是应用层中每一层后输入中input_placeholders的输出


    * this is necessary since spectralnet takes multiple inputs and performs special computations on the
      orthonorm layer
      这是必要的，因为SpectralNet接受多个输入并在正交范数层上执行特殊计算
    '''
    # 定义输出为一个字典类型
    outputs = dict()
    # 将inputs的值赋值给outputs
    outputs = inputs
    # for key in inputs:
    #     outputs[key]=inputs[key]
    # layers是一个数组，数组的元素为字典
    # layers = [{'type':'Orthonorm', 'name':'orthonorm'}]
    for layer in layers:
        # check for l2_reg argument
        # 字典.get()  返回指定键的值
        l2_reg = layer.get('l2_reg')
        if l2_reg:
            l2_reg = l2(layer['l2_reg'])

        # create the layer
        # 创建网络层
        if layer['type'] == 'softplus_reg':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001), name=layer.get('name'))
        elif layer['type'] == 'softplus':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'softmax':
            l = Dense(layer['size'], activation='softmax', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'tanh':
            l = Dense(layer['size'], activation='tanh', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'relu':
            l = Dense(layer['size'], activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'selu':
            l = Dense(layer['size'], activation='selu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'Conv2D':
            l = Conv2D(layer['channels'], kernel_size=layer['kernel'], activation='relu', data_format='channels_last', kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'BatchNormalization':
            l = BatchNormalization(name=layer.get('name'))
        elif layer['type'] == 'MaxPooling2D':
            l = MaxPooling2D(pool_size=layer['pool_size'], data_format='channels_first', name=layer.get('name'))
        elif layer['type'] == 'Dropout':
            l = Dropout(layer['rate'], name=layer.get('name'))
        elif layer['type'] == 'Flatten':
            l = Flatten(name=layer.get('name'))

        # 若传入的网络层的type为Othonorm
        elif layer['type'] == 'Orthonorm':
            # 则调用Orthonorm方法
            l = Orthonorm(outputs, name=layer.get('name'))
        elif layer['type'] == 'fft_layer':
            # 则调用Dense_fft方法
            l = fft_layer(outputs, output_num = layer.get('output_num'), name = layer.get('name'))
        else:
            raise ValueError("Invalid layer type '{}'".format(layer['type']))

        # apply the layer to each input in inputs
        # for k in outputs:
        outputs=l(outputs)

    return outputs
