import tensorflow as tf


def split_heads_2d(inputs, Nh):
    s = inputs_shape[:-1]
    ret_shape = s + [Nh, s // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(inputs):
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
    a, b = transposed.shape[-2:]
    ret_shape = transposed.shape[:-2] + [a * b]
    return tf.reshape(transposed, ret_shape)


def compute_flat_qkv(inputs, dk, dv, Nh):
    N, H, W, _ = inputs.shape
    qkv = tf.layers.conv2d(inputs, 2 * dk + dv, 1)
    q, k, v = tf.split(qkv, [dk, dk, dv], axis=3)
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)
    dkh = dk // Nh
    q *= dkh ** -0.5
    flat_q = tf.reshape(q, [N, Nh, H * W, dk])
    flat_k = tf.reshape(k, [N, Nh, H * W, dk])
    flat_v = tf.reshape(v, [N, Nh, H * W, dv])
    return flat_q, flat_k, flat_v


def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
    rel_logits = tf.einsum('bhxyd,md−>bhxym', q, rel_k)
    rel_logits = tf.reshape(rel_logits, [−1, Nh * H, W, 2 * W−1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = tf.reshape(rel_logits, [−1, Nh, H, W, W])
    rel_logits = tf.expand dims(rel_logits, axis=3)
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    rel_logits = tf.reshape(rel_logits, [−1, Nh, H * W, H * W])
    return rel_logits


def relative_logits(q):
    dk = q.shape[-1]
    key_rel_w = tf.get_variable(
        'key_rel_w', shape=(2 * W−1, dk),
        initializer=tf.random_normal_initializer(dk **−0.5)
    )
    rel_logits_w = relative_logits_1d(
        q, key_rel_w, H, W, [0, 1, 2, 4, 3, 5]
    )
    key_rel_h = tf.get variable(
        'key_rel_h', shape=(2 * H−1, dk),
        initializer=tf.random_normal_initializer(dk **−0.5)
    )
    rel_logits_h = relative_logits_1d(
        tf.transpose(q, [0, 1, 3, 2, 4]),
        key_rel_h, W, H, [0, 1, 4, 2, 5, 3]
    )
    return rel_logits_h, rel_logits_w


def rel_to_abs(x):
    B, Nh, L, = x.shape
    col_pad = tf.zeros((B, Nh, L, 1))
    x = tf.concat([x, col_pad], axis=3)
    flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
    flat_pad = tf.zeros((B, Nh, L−1))
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
    final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L−1])
    final_x = final_x[:, :, :L, L−1:]
    return final_x


def augmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
    conv_out = tf.layers.conv2d(X, Fout − dv, k)
    flat_q, flat_k, flat_v = compute_flat_qkv(X, dk, dv)
    logits = tf.matmul(flat_ q, flat_k, transpose_b=True)
    if relative:
        h_rel_logits, w_rel_logits = relative_logits(q)
        logits += h_rel_logits
        logits += w_rel_logits
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flat_v)
    attn_out = tf.reshape(v, [B, Nh, H, W, dv // Nh])
    attn_out = combine_heads_2d(v)
    attn_out = tf.layers.conv2d(attn_out, dv, 1)
    return tf.concat([conv_out, attn_out], axis=3)
