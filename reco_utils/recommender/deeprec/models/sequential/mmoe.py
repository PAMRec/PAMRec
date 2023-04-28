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
# import tensorflow_ranking as tfr
import os
import numpy as np

__all__ = ["MMoEModel_original"]


class MMoEModel_original(SequentialBaseModel):
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
        self.data_loss = self._compute_data_loss()
        self.auxiliary_data_loss = self._compute_auxiliary_data_loss()
        self.regular_loss = self._compute_regular_loss()

        self.loss = self.data_loss + self.regular_loss + self.auxiliary_data_loss
        return self.loss

    def _compute_auxiliary_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.valid_logit, [-1]),
                    labels=tf.reshape(self.iterator.labels_play, [-1]),
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
        return tf.multiply(0.5, data_loss)  # 1 0.1 0.5 0.4 0.6 0.7 0.8 0.9 0.2 0.3

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """

        return super(MMoEModel_original, self)._build_train_opt()


    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(MMoEModel_original, self)._build_embedding()
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

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(MMoEModel_original, self)._lookup_from_embedding()

        # temp = tf.cast(self.iterator.item_duration_history, dtype=tf.int32)
        # self.history_play_embedding = tf.nn.embedding_lookup(self.play_lookup, temp)
        # temp2 = tf.cast(self.iterator.satisfied_duration_history, dtype=tf.int32)
        # self.satisfied_history_play_embedding = tf.nn.embedding_lookup(self.play_lookup, temp2)



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

            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return logit

    def _build_seq_graph(self):
        """The main function to create clsr model.
        
        Returns:
            obj:the output of clsr section.
        """
        hparams = self.hparams
        with tf.variable_scope("clsr"):
            # self.history_play_embedding = self.iterator.item_play_value_history



            hist_input = tf.concat(
                [self.satisfied_item_history_embedding, self.satisfied_cate_history_embedding], 2 # change
            )

            self.mask = self.iterator.satisfied_mask # change
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            # valid play
            hist_valid_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2  # change
            )
            hist_valid_input = hist_valid_input# + self.history_play_embedding
            self.valid_mask = self.iterator.mask # change
            self.valid_real_mask = tf.cast(self.valid_mask, tf.float32)
            self.valid_sequence_length = tf.reduce_sum(self.valid_mask, 1)


            with tf.variable_scope("long_term"):
                # att_outputs_long = self._attention_fcn(self.user_long_embedding, hist_input)
                att_outputs_long = self._attention_fcn(self.target_item_embedding, hist_input, self.mask)
                self.att_fea_long = tf.reduce_sum(att_outputs_long, 1)
                tf.summary.histogram("att_fea_long", self.att_fea_long)
            with tf.variable_scope("short_term"):
                # att_outputs_short = self._attention_fcn(self.user_short_embedding, hist_valid_input, self.valid_mask)
                att_outputs_short = self._attention_fcn(self.target_item_embedding, hist_valid_input, self.valid_mask)
                self.att_fea_short = tf.reduce_sum(att_outputs_short, 1)
                tf.summary.histogram("att_fea2", self.att_fea_short)

            # model_output, self.valid_output = self._mmoe_layer(tf.concat([self.att_fea_long, self.att_fea_short, self.target_item_embedding], 1))
            model_output, self.valid_output = self._mmoe_layer(tf.concat([self.att_fea_long, self.att_fea_short, self.target_item_embedding], 1))

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
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(
        #     sess,
        #     ui_type="readline")
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

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
                # self.order_loss,
                # self.discrepancy_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def step_train(self, step, step_result):
        (_, _, step_loss, step_data_loss, step_regular_loss, step_auxiliary_data_loss,
         summary) = step_result
        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            self.writer.add_summary(summary, step)
        if step % self.hparams.show_step == 0:
            print(
                "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, auxiliary_data_loss: {3:.4f}".format(
                    step, step_loss, step_data_loss, step_auxiliary_data_loss
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
        # tf.summary.scalar("contrastive_loss", self.contrastive_loss)
        # tf.summary.scalar("discrepancy_loss", self.discrepancy_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged
