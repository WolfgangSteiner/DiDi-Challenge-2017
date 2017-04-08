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


def log_loss(y_true, y_pred):
    '''
    Compute the softmax log loss.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.
    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://github.com/pierluigiferrari/ssdkeras/blob/master/keras_ssd_loss.py
    '''
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.log(y_pred))
    return log_loss


def multitask_loss(y_true, y_pred):
    num_classes = 1
    nx = 32
    ny = 32
    num_parameters = 6
    y_shape = (-1,nx*ny,num_classes+num_parameters)
    y_true = tf.reshape(y_true, y_shape)
    y_pred = tf.reshape(y_pred, y_shape)
    positives = y_true[:,:,0]

    # number of positive boxes per example
    # size (batch_size)
    n_positives = tf.reduce_sum(positives,axis=-1)

    y_true_conf = y_true[:,:,0]
    y_pred_conf = tf.sigmoid(y_pred[:,:,0])

    confidence_loss = -y_true_conf * tf.log(tf.maximum(1e-15, y_pred_conf)) \
        - (1 - y_true_conf) * tf.log(tf.maximum(1e-15, 1.0 - y_pred_conf))
    confidence_loss = tf.reduce_sum(confidence_loss, axis=-1)

    localization_loss = loc_loss(y_true, y_pred) * positives
    localization_loss = tf.reduce_sum(localization_loss, axis=-1)


    total_loss = (1.0 / tf.maximum(n_positives, 1)) * (confidence_loss + localization_loss)

#    zero_loss = tf.zeros_like(localization_loss)
#    total_loss = tf.where(tf.equal(n_positives, 0), zero_loss, total_loss)

    return total_loss

    #loc_loss = smooth_L1_loss(y_true_params, y_pred_params)
    #confidence_loss = log_loss(y_true[:,:,0], y_pred[:,:,0])
    #if n_positives == 0:
    #    return tf.zeros_like(y_true[:,0,0])
    #else:
    #    return
