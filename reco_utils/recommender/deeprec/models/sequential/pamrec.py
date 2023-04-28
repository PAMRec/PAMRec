# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
# from tensorflow.nn import dynamic_rnn
from tensorflow.compat.v1.nn import dynamic_rnn
# from tensorflow.nn import dynamic_rnn

from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
import tensorflow_ranking as tfr
import os
import numpy as np

__all__ = ["PAMREC"]



class PAMRECModel(SequentialBaseModel):
    def _mmoe_layer(self, x):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        experts_output = []
        for i in range(hparams.expert_num):
            scope_name = "expert_{}".format(i)
            experts_output.append(tf.expand_dims(self._fcn_transform_net(x, hparams.expert_layer_sizes, scope_name), 1)) # batch_size * 1 * dim
        experts = tf.concat(experts_output, 1)  # batch_size * experts_num * dim

        gate_main = tf.expand_dims(self._fcn_transform_net(x, hparams.gate_layer_sizes, "gate_main"), 1)  # batch_size * 1 * experts_num
        gate_sub = tf.expand_dims(self._fcn_transform_net(x, hparams.gate_layer_sizes, "gate_sub"), 1)  # batch_size * 1 * experts_num
        main = tf.matmul(gate_main, experts)
        sub = tf.matmul(gate_sub, experts)
        main = tf.squeeze(main, [1])
        sub = tf.squeeze(sub, [1])
        return main, sub



    def _get_loss(self):
        """Make loss function, consists of data loss, regularization loss, contrastive loss and discrepancy loss
        
        Returns:
            obj: Loss value
        """
        tf.summary.histogram("logit", tf.nn.sigmoid(self.logit))
        tf.summary.histogram("valid_logit", tf.nn.sigmoid(self.valid_logit))
        self.data_loss = self._compute_data_loss()
        self.auxiliary_data_loss = self._compute_auxiliary_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.order_loss = self._compute_order_loss()
        # self.order_loss = tf.constant(0.0)
        self.loss = self.data_loss + self.regular_loss + self.auxiliary_data_loss + self.order_loss
        return self.loss

    def _compute_order_loss(self):
        xilidu_logit = self._fcn_net(self.model_output, self.hparams.layer_sizes, scope="xilidu_logit_fcn")
        # self._fcn_transform_net(self.new_long, self.hparams.gate_layer_sizes, "gate_sub")
        xilidu_logit = tf.reshape(xilidu_logit, (-1, 5))
        xilidu_logit = self._activate(xilidu_logit, "sigmoid")
        labels = tf.reshape(self.iterator.plays, (-1, 5))
        mean_loss = tfr.losses._approx_ndcg_loss(labels, xilidu_logit, reduction=tf.compat.v1.losses.Reduction.MEAN)
        # mean_loss = tf.ones_like(labels)
        # mean_loss = tf.reduce_mean(mean_loss, axis=-1)
        return tf.multiply(self.hparams.discrepancy_loss_weight, mean_loss)

    def _compute_auxiliary_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.valid_logit, [-1]),
                    labels=tf.reshape(self.iterator.labels_play, [-1]),
                )
            )
        elif self.hparams.loss == "log_loss":
            # raise Exception("error")
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.pred, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.valid_logit, (-1, group))
            labels = tf.reshape(self.iterator.labels_play, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        return tf.multiply(self.hparams.fuzhu_weight, data_loss)  # 1 0.1 0.5 0.4 0.6 0.7 0.8 0.9 0.2 0.3

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """

        return super(PAMRECModel, self)._build_train_opt()


    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(PAMRECModel, self)._build_embedding()
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            self.user_long_lookup = tf.get_variable(
                name="user_long_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.user_short_lookup = tf.get_variable(
                name="user_short_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.play_lookup = tf.get_variable(
                name="play_lookup",
                shape=[10, 40],
                dtype=tf.float32,
            )
            self.position_embedding_lookup = tf.get_variable(
                name="position_embedding",
                shape=[self.hparams.max_seq_length, (self.item_embedding_dim + self.cate_embedding_dim) * 2], # test add target item
                # shape=[self.hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim], # test add target item
                dtype=tf.float32,
            )


    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(PAMRECModel, self)._lookup_from_embedding()

        self.position_embedding = tf.nn.embedding_lookup(
            self.position_embedding_lookup,
            tf.tile(tf.expand_dims(tf.range(self.hparams.max_seq_length), 0),
                    [tf.shape(self.item_history_embedding)[0], 1])
        )
        self.position_embedding = self._dropout(
            self.position_embedding, keep_prob=self.embedding_keeps
        )
        self.user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.iterator.users
        )
        tf.summary.histogram("user_long_embedding_output", self.user_long_embedding)

        self.user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.iterator.users
        )
        tf.summary.histogram("user_short_embedding_output", self.user_short_embedding)

        involved_users = tf.reshape(self.iterator.users, [-1])
        self.involved_users, _ = tf.unique(involved_users)
        self.involved_user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_long_embedding)
        self.involved_user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_short_embedding)

        # dropout after embedding
        self.user_long_embedding = self._dropout(
            self.user_long_embedding, keep_prob=self.embedding_keeps
        )
        self.user_short_embedding = self._dropout(
            self.user_short_embedding, keep_prob=self.embedding_keeps
        )


    def _build_graph(self):
        """The main function to create sequential models.

        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()

            # add
            self.valid_logit = self._fcn_net(self.valid_output, hparams.layer_sizes, scope="valid_logit_fcn")
            # self.valid_logit = self._activate(self.valid_logit, "sigmoid", layer_idx=-1)
            self.model_output = model_output
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            # logit = self._activate(logit, "sigmoid", layer_idx=-1)
            # logit = logit * self.valid_logit
            self._add_norm()
            return logit






    def _build_seq_graph(self):
        """The main function to create pamrec model.
        
        Returns:
            obj:the output of pamrec section.
        """
        hparams = self.hparams
        # self.mask_satisfied()
        with tf.variable_scope("pamrec"):
            # self.history_play_embedding = self.iterator.item_play_value_history
            if self.hparams.fine_tune is False:
                self.mask = self.iterator.mask  # batch_size * seq_len
                self.valid_mask = self.iterator.mask  # change
                self.satisfied_mask = self.iterator.item_satisfied_value_history
                self.valid_mask_satisfied = self.iterator.item_satisfied_value_history
                print("False")
            else:
                self.mask = self.mask_gumbel
                self.valid_mask = self.mask_gumbel
                self.satisfied_mask = self.satisfied_mask_gumbel
                self.valid_mask_satisfied = self.satisfied_mask_gumbel
                print("True")

            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)  # batch_size
            self.history_embedding = tf.concat(  # batch_size * hist_len * embedding_len
                [self.item_history_embedding,
                 self.cate_history_embedding,
                 tf.tile(tf.expand_dims(self.target_item_embedding, -2), [1, self.item_history_embedding.shape[-2].value, 1])
                 ], 2 # add
            )
            self.history_embedding = self.history_embedding + self.position_embedding
            seq = self.sasrec(
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
            )
            hist_input = seq
            tf.summary.histogram("SASRec_outputs", seq)

            # valid play
            self.valid_real_mask = tf.cast(self.valid_mask, tf.float32)
            self.valid_sequence_length = tf.reduce_sum(self.valid_mask, 1)

            # satisfied
            self.satisfied_real_mask = tf.cast(self.satisfied_mask, tf.float32)

            with tf.variable_scope("new_long"):
                score = self._fcn_transform_net(hist_input, [20, 1], "score_1")
                boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
                att_fnc_output = tf.squeeze(score, -1)
                mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
                att_weights = tf.nn.softmax(
                    tf.where(boolean_mask, att_fnc_output, mask_paddings),
                    name="att_weights",
                )
                output = hist_input * tf.expand_dims(att_weights, -1)
                self.new_long = tf.reduce_sum(output, 1)
                tf.summary.histogram("new_long", self.new_long)


            self.valid_real_mask_satisfied = tf.cast(self.valid_mask_satisfied, tf.float32)
            self.valid_weak_mask = tf.constant([1], tf.float32) - self.valid_real_mask_satisfied
            boolean_mask = tf.equal(self.valid_mask, tf.ones_like(self.valid_mask))
            mask_paddings = tf.zeros_like(self.valid_mask, tf.float32)
            self.valid_weak_mask = tf.where(boolean_mask, self.valid_weak_mask, mask_paddings)


            with tf.variable_scope("new_distill"):
                new_distill = self._attention_fcn(self.new_long, hist_input, self.valid_weak_mask)
                self.new_distiall = tf.reduce_sum(new_distill, 1)
                tf.summary.histogram("new_distiall", self.new_distiall)





            with tf.variable_scope("long_term"):
                att_outputs_long = self._attention_fcn(self.user_long_embedding, hist_input, mask=self.satisfied_mask)
                self.att_fea_long = tf.reduce_sum(att_outputs_long, 1)
                tf.summary.histogram("att_fea_long", self.att_fea_long)


            with tf.variable_scope("short_term"):
                att_outputs_short = self._attention_fcn(self.user_short_embedding, hist_input, self.valid_mask)
                self.att_fea_short = tf.reduce_sum(att_outputs_short, 1)
                tf.summary.histogram("att_fea2", self.att_fea_short)

            model_output, self.valid_output = self._mmoe_layer(self.new_long)

            model_output = tf.concat([model_output, self.target_item_embedding], 1)
            self.valid_output = tf.concat([self.valid_output, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

    def _fcn_transform_net(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                nn_output = hidden_nn_layers[-1]
                return nn_output

    def _attention_fcn(self, query, user_embedding, mask=None):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        if mask is None:
            mask = self.mask
        hparams = self.hparams
        with tf.variable_scope("attention_fcn"):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(mask, tf.ones_like(mask))

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output

    def train(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """

        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.regular_loss,
                self.auxiliary_data_loss,
                self.order_loss,
                # self.discrepancy_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def step_train(self, step, step_result):
        (_, _, step_loss, step_data_loss, step_regular_loss, step_auxiliary_data_loss, order_loss,
         summary) = step_result
        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            self.writer.add_summary(summary, step)
        if step % self.hparams.show_step == 0:
            print(
                "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, auxiliary_data_loss: {3:.4f}, order_loss: {4:.4f}".format(
                    step, step_loss, step_data_loss, step_auxiliary_data_loss, order_loss
                )
            )

    def batch_train(self, file_iterator, train_sess):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_regular_loss = 0
        epoch_auxiliary_data_loss = 0
        # epoch_order_loss = 0
        for batch_data_input in file_iterator:
            if batch_data_input:
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, step_regular_loss, step_auxiliary_data_loss, summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                epoch_data_loss += step_data_loss
                epoch_regular_loss += step_regular_loss
                epoch_auxiliary_data_loss += step_auxiliary_data_loss
                # epoch_order_loss += step_order_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, auxiliary_data_loss: {3:.4f}".format(
                            step, step_loss, step_data_loss, step_auxiliary_data_loss
                        )
                    )

        return epoch_loss

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("order_loss", self.order_loss)
        tf.summary.scalar("auxiliary_data_loss", self.auxiliary_data_loss)
        # tf.summary.scalar("contrastive_loss", self.contrastive_loss)
        # tf.summary.scalar("discrepancy_loss", self.discrepancy_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    def sasrec(self, inputs, sequence_length):
        self.seq = inputs
        for i in range(2):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = self.multihead_attention(queries=self.normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=(self.item_embedding_dim + self.cate_embedding_dim) * 2, # add
                                                    # num_units=self.item_embedding_dim + self.cate_embedding_dim, # add
                                                    num_heads=1,
                                                    dropout_rate=self.embedding_keeps,
                                                    causality=False,
                                                    scope="self_attention")


                # Feed forward
                # self.seq = self._fcn_transform_net(self.normalize(self.seq), [self.item_embedding_dim + self.cate_embedding_dim,
                #                                               self.item_embedding_dim + self.cate_embedding_dim], "mlp")
                self.seq = self.feedforward(self.normalize(self.seq), num_units=[(self.item_embedding_dim + self.cate_embedding_dim) * 2,
                                                                                 (self.item_embedding_dim + self.cate_embedding_dim) * 2])

                # self.seq = self.feedforward(self.normalize(self.seq),
                #                             num_units=[self.item_embedding_dim + self.cate_embedding_dim,
                #                                        self.item_embedding_dim + self.cate_embedding_dim])

                # self.seq *= mask

        return self.seq

    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    # dropout_rate=0.2,
                    # is_training=True,
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # Residual connection
            outputs += inputs

            # Normalize
            # outputs = normalize(outputs)

        return outputs

    def _fcn_transform_net_transformer(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                    )
                    # activation = hparams.activation[idx]
                    # curr_hidden_nn_layer = self._active_layer(
                    #     logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    # )
                    curr_hidden_nn_layer = self._dropout(curr_hidden_nn_layer, keep_prob=self.embedding_keeps)
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                nn_output = hidden_nn_layers[-1] + model_output
                return nn_output

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        return outputs

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None,
                            with_qk=False):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]


            self.Q_timeaware_embedding_lookup = tf.get_variable(
                name="Q_timeaware_embedding",
                shape=[10, num_units * num_units],
                dtype=tf.float32,
            )
            self.K_timeaware_embedding_lookup = tf.get_variable(
                name="K_timeaware_embedding",
                shape=[10, num_units * num_units],
                dtype=tf.float32,
            )
            self.V_timeaware_embedding_lookup = tf.get_variable(
                name="V_timeaware_embedding",
                shape=[10, num_units * num_units],
                dtype=tf.float32,
            )
            self.Q_timeaware_embedding = tf.reshape(tf.nn.embedding_lookup(
                self.Q_timeaware_embedding_lookup,
                tf.cast(self.iterator.item_loop_times_history, dtype=tf.int32)
            ), [-1,queries.shape[1], num_units, num_units])
            self.K_timeaware_embedding = tf.reshape(tf.nn.embedding_lookup(
                self.K_timeaware_embedding_lookup,
                tf.cast(self.iterator.item_loop_times_history, dtype=tf.int32)
            ), [-1, keys.shape[1], num_units, num_units])
            self.V_timeaware_embedding = tf.reshape(tf.nn.embedding_lookup(
                self.V_timeaware_embedding_lookup,
                tf.cast(self.iterator.item_loop_times_history, dtype=tf.int32)
            ), [-1, keys.shape[1], num_units, num_units])
            Q = tf.squeeze(tf.matmul(tf.expand_dims(queries, -2), self.Q_timeaware_embedding), -2)
            K = tf.squeeze(tf.matmul(tf.expand_dims(keys, -2), self.K_timeaware_embedding), -2)
            V = tf.squeeze(tf.matmul(tf.expand_dims(keys, -2), self.V_timeaware_embedding), -2)

            tf.summary.histogram(
                "Q", Q
            )
            tf.summary.histogram(
                "K", K
            )
            tf.summary.histogram(
                "V", V
            )
            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # array([array([0.00000000e+00, 4.09545455e-02, 1.28786778e-01, 3.27894289e-01,
            # 7.66666667e-01, 1.05850847e+00, 1.18596610e+00, 1.53888889e+00,
            # 2.13000000e+00, 3.18951300e+03])], dtype=object)




            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)









            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)


            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = self.mask
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)


            # Dropouts
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)





            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries
        return outputs

    def _mlp(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = (
                        tf.tensordot(
                            hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                        )
                        + curr_b_nn_layer
                    )
                    activation = hparams.activation[idx]
                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )
                    if idx != 1:
                        curr_hidden_nn_layer = self._active_layer(
                            logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                        )

                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size
                return hidden_nn_layers[-1]