import tensorflow as tf


def smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.
    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    References:
        https://github.com/pierluigiferrari/ssdkeras/blob/master/keras_ssd_loss.py
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss)


def loc_loss(y_true, y_pred):
    '''
    Calculates the smooth l1 loss between the parameters of ground truth and prediction.
    Only the entries of the tensor that belong to a positive sample are considered.

    Returns:
       Localization loss, a tensor of shape (batch, n_boxes_total)
    '''
    residual = (y_true[:,:,1:] - y_pred[:,:,1:])
    absolute_loss = tf.abs(residual)
    square_loss = 0.5 * (residual)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def safe_log(y, eps=1e-15):
    return tf.log(tf.maximum(eps, y))


def log_loss(y_true, y_pred):
    '''
    Calculate the log loss between y_true and y_pred. y_pred is considered to be the output
    of a sigmoid activation function.
    '''
    eps = 1e-15
    return -y_true * safe_log(y_pred) - (1.0 - y_true) * safe_log(1.0 - y_pred)


def multitask_loss(y_true, y_pred):
    alpha = 0.125
    nx = 32
    ny = 32
    num_classes = 1
    num_parameters = 6
    y_shape = (-1,nx*ny,num_classes+num_parameters)
    y_true = tf.reshape(y_true, y_shape)
    y_pred = tf.reshape(y_pred, y_shape)
    positives = y_true[:,:,0]
    negatives = 1.0 - positives

    y_true_conf = y_true[:,:,0]
    y_pred_conf = tf.sigmoid(y_pred[:,:,0])
    n_positives = tf.reduce_sum(positives,axis=-1)
    n_neg_examples_to_keep = tf.to_int32(tf.reduce_sum(n_positives) * 3)

    # calculate the log loss
    positive_classification_loss = -positives * safe_log(y_pred_conf)
    negative_classification_loss = -negatives * safe_log(1.0 - y_pred_conf)
    negative_classification_loss_1d = tf.reshape(negative_classification_loss, (-1,))

    values, indices = tf.nn.top_k(negative_classification_loss_1d, n_neg_examples_to_keep, False)
    negative_classification_loss_1d = tf.scatter_nd(indices, values, tf.shape(negative_classification_loss_1d))
    negative_classification_loss = tf.reshape(negative_classification_loss, (-1,nx*ny))

    confidence_loss = positive_classification_loss + negative_classification_loss
    confidence_loss = tf.reduce_sum(confidence_loss, axis=-1)

    localization_loss = loc_loss(y_true, y_pred) * positives
    localization_loss = tf.reduce_sum(localization_loss, axis=-1)

    total_loss = (1.0 / tf.maximum(n_positives, 1)) * (confidence_loss + alpha * localization_loss)

    return total_loss
