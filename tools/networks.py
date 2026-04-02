"""
Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation

This script contains the `build_ce_dnn` function, which defines
and trains the DNN-based channel estimator using TensorFlow.

TODO:
Complete the `build_ce_dnn` function. You need to define the input/output
placeholders and realize the network architecture and loss function.
"""

import numpy as np
import numpy.linalg as la
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tools.shrinkage as shrinkage
from .train import load_trainable_vars, save_trainable_vars
from .raputil import sample_gen
from tensorflow.keras.layers import Dense


def build_ce_dnn(K, SNR, savefile, learning_rate=1e-3, training_epochs=2000,
                 batch_size=50, nh1=500, nh2=250, test_flag=False, cp_flag=True):
    n_input = 2 * K + 2 * K  # yp and xp as input
    n_output = 2 * K
                     
    # please fill in the blank in the following codes
                     
    # filled starter TODOs only
    # 把輸入 placeholder 定義成 float64，是為了讓它和後面 Dense layer 的 dtype='float64' 完全一致，
    # 避免在 TensorFlow / Keras 環境下發生 float32 與 float64 混用的型別錯誤
    nn_input = tf.placeholder(tf.float64, (None, n_input), name='nn_input')

    # H_true 是真實通道頻域響應的標籤，同樣使用 float64，
    # 目的是讓 loss 計算時 H_out 與 H_true 保持相同精度與型別
    H_true = tf.placeholder(tf.float64, (None, n_output), name='H_true')

    # 兩層 hidden layer + 一層輸出層對應題目要求的 DNN estimator
    # 保留原本 MLP 架構，hidden layer 使用 ReLU，輸出層不加 activation，因為通道估測是連續值回歸問題，不是分類問題
    # 同時顯式指定 dtype='float64'，與 placeholder 保持一致
    dense1 = Dense(nh1, activation='relu', dtype='float64')
    dense2 = Dense(nh2, activation='relu', dtype='float64')
    output_layer = Dense(n_output, activation=None, dtype='float64')

    # 依序通過兩層 hidden layer，最後輸出 2K 維實數向量，對應通道頻響的實部與虛部。
    tmp = dense1(nn_input)
    tmp = dense2(tmp)
    H_out = output_layer(tmp)

    # Define loss and optimizer, minimize the l2 loss
    # 這裡使用 l2 loss 來衡量估測通道與真實通道之間的誤差，符合通道估測是最小化均方誤差的目標
    # H_true[:, :n_output] 的寫法保留原本 starter code 的輸出對齊方式
    loss_ = tf.nn.l2_loss(H_out - H_true[:, :n_output])
    global_step = tf.Variable(0, trainable=False)
    decay_steps, lr_decay = 20000, 0.1
    lr_ = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay, name='lr')

    # 使用 AdamOptimizer 進行訓練，搭配 exponential decay 讓學習率隨訓練逐步下降，以提升收斂穩定性
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_).minimize(
        loss_, global_step, var_list=tf.trainable_variables()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    log = str(state.get('log', ''))
    print(log)

    if test_flag:
        return sess, nn_input, H_out

    test_step = 5
    loss_history = []
    save = {}  # for the best model

    val_ls, val_labels, val_Yp, val_Xp = sample_gen(batch_size * 100, SNR, training_flag=False, CP_flag=cp_flag)
    for epoch in range(training_epochs + 1):
        train_loss = 0.
        for m in range(20):
            batch_ls, batch_labels, Yp, Xp = sample_gen(batch_size, SNR, training_flag=True, CP_flag=cp_flag)
            sample = np.concatenate((Yp, Xp), axis=1)  # (bs, 4K)

            # 直接把 Yp 與 Xp 串接成網路輸入，對應題目中「pilot 接收訊號 + pilot 已知符號」共同作為 DNN 輸入的設定
            _, loss = sess.run([optimizer, loss_], feed_dict={nn_input: sample, H_true: batch_labels})
            train_loss += loss
        sys.stdout.write('\repoch={epoch:<6d} loss={loss:.9f} on train set'.format(epoch=epoch, loss=train_loss))
        sys.stdout.flush()

        # validation
        if epoch % test_step == 0:
            sample = np.concatenate((val_Yp, val_Xp), axis=1)  # (bs, 4K)
            loss = sess.run(loss_, feed_dict={nn_input: sample, H_true: val_labels})
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                # 把目前 validation loss 最好的權重暫存起來，目的是避免最後一個 epoch 過擬合，並在訓練結束後回復最佳模型
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
            print("\nepoch={epoch:<6d} loss={loss:.9f} (best={best:.9f}) on test set".format(epoch=epoch, loss=loss, best=loss_best))

    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} iterations'.format(
        loss=loss, i=epoch, best=loss_best, j=loss_history.argmin() * test_step)

    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    print("optimization finished")

    return sess, nn_input, H_out
