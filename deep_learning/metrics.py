import keras.backend as K


def _prepare_y(y_true, y_pred):
    shape = K.int_shape(y_pred)[1]
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, shape)
    y_true = K.cast(y_true, dtype='int32')
    y_true = K.squeeze(y_true, axis=-1)
    y_true = K.one_hot(y_true, shape)
    return y_true, y_pred


def _get_report(y_true, y_pred):
    TP = K.sum(y_true * y_pred)
    FP = K.sum((K.ones_like(y_true) - y_true) * y_pred)
    FN = K.sum(y_true * (K.ones_like(y_true) - y_pred))
    return TP, FP, FN


def f1_micro(y_true, y_pred):
    y_true, y_pred = _prepare_y(y_true, y_pred)
    TP, FP, FN = _get_report(y_true, y_pred)
    recall = TP / (TP + FN + K.epsilon())
    precision = TP / (TP + FP + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
