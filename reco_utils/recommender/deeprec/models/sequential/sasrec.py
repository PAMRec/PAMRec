# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
# from tensorflow.contrib.rnn import GRUCell, LSTMCell
# from tensorflow.nn import dynamic_rnn

__all__ = ["SASRecModel"]


class SASRecModel(SequentialBaseModel):
    """
    SASRec
    """
    def _build_embedding(self):
        super(SASRecModel, self)._build_embedding()
        with tf.variable_scope("embedding", initializer=self.initializer):
            if self.hparams.add_feature:
                self.position_embedding_lookup = tf.get_variable(
                    name="position_embedding",
                    shape=[self.hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim * 2],
                    dtype=tf.float32,
                )
            else:
                self.position_embedding_lookup = tf.get_variable(
                    name="position_embedding",
                    shape=[self.hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim],
                    dtype=tf.float32,
                )

    def _lookup_from_embedding(self):
        super(SASRecModel, self)._lookup_from_embedding()
        self.position_embedding = tf.nn.embedding_lookup(
            self.position_embedding_lookup,
            tf.tile(tf.expand_dims(tf.range(self.hparams.max_seq_length), 0),
                    [tf.shape(self.satisfied_item_history_embedding)[0], 1])
        )
        self.position_embedding = self._dropout(
            self.position_embedding, keep_prob=self.embedding_keeps
        )

    def _build_seq_graph(self):
        """The main function to create GRU4Rec model.

        Returns:
            obj:the output of GRU4Rec section.
        """
        with tf.variable_scope("sasrec"):
            final_state = self._build_sasrec()
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
        return model_output

    def _build_sasrec(self):
        self.mask = self.iterator.satisfied_mask  # batch_size * seq_len
        self.sequence_length = tf.reduce_sum(self.mask, 1)  # batch_size
        self.history_embedding = tf.concat(  # batch_size * hist_len * embedding_len
            [self.satisfied_item_history_embedding, self.satisfied_cate_history_embedding], 2
        )
        self.history_embedding = self.history_embedding + self.position_embedding

        seq = self.sasrec(
            inputs=self.history_embedding,
            sequence_length=self.sequence_length,
        )
        tf.summary.histogram("SASRec_outputs", seq)

        right = tf.cast(self.sequence_length - tf.ones_like(self.sequence_length), tf.int32)
        right = tf.expand_dims(right, 1)
        left = tf.range(tf.shape(self.sequence_length)[0])
        left = tf.expand_dims(left, 1)
        ind_tensor = tf.concat([left, right], -1)
        final_state = tf.gather_nd(seq, ind_tensor)
        return final_state


    def sasrec(self, inputs, sequence_length):
        self.seq = inputs
        for i in range(2):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = self.multihead_attention(queries=self.normalize(self.seq),
                                               keys=self.seq,
                                               # num_units=self.item_embedding_dim + self.cate_embedding_dim,
                                               num_units=self.history_embedding.shape[-1].value,
                                               num_heads=1,
                                               dropout_rate=self.embedding_keeps,
                                               causality=False,
                                               scope="self_attention")


                # Feed forward
                # self.seq = self._fcn_transform_net(self.normalize(self.seq), [self.item_embedding_dim + self.cate_embedding_dim,
                #                                               self.item_embedding_dim + self.cate_embedding_dim], "mlp")
                # self.seq = self.feedforward(self.normalize(self.seq), num_units=[self.item_embedding_dim + self.cate_embedding_dim,
                #                                               self.item_embedding_dim + self.cate_embedding_dim])
                self.seq = self.feedforward(self.normalize(self.seq),
                                            num_units=[self.history_embedding.shape[-1].value,
                                                       self.history_embedding.shape[-1].value])
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

            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

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

            # Query Masking
            # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            # query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            # outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = self._dropout(outputs, keep_prob=self.embedding_keeps)
            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries
        return outputs
            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

        # if with_qk:
        #     return Q, K
        # else:
        #     return outputs

    # def _build_seq_graph(self):
    #     """The main function to create GRU4Rec model.
        
    #     Returns:
    #         obj:the output of GRU4Rec section.
    #     """
    #     with tf.variable_scope("sasrec"):
    #         self.mask = self.iterator.satisfied_mask
    #         self.sequence_length = tf.reduce_sum(self.mask, 1)
    #         self.history_embedding = tf.concat(
    #             [self.satisfied_item_history_embedding, self.satisfied_cate_history_embedding], 2
    #         )

    #         # final_state = self._build_lstm()
    #         final_state = self._build_sasrec()
    #         model_output = tf.concat([final_state, self.target_item_embedding], 1)
    #         tf.summary.histogram("model_output", model_output)
    #         return model_output
    #         self.seq, item_emb_table = embedding(input_seq,
    #                                              vocab_size=self.itemnum,
    #                                              num_units=self.hidden_units,
    #                                              zero_pad=False,
    #                                              scale=True,
    #                                              l2_reg=self.l2_emb,
    #                                              scope="input_embeddings",
    #                                              with_t=True,
    #                                              reuse=None
    #                                              )

    # def _build_sasrec(self):
    #     """Apply SASRec for modeling.

    #     Returns:
    #         obj: The output of SASRec section.
    #     """
    #     with tf.name_scope("sas"):
    #         self.mask = self.iterator.satisfied_mask
    #         self.sequence_length = tf.reduce_sum(self.mask, 1)
    #         self.history_embedding = tf.concat(
    #             [self.satisfied_item_history_embedding, self.satisfied_cate_history_embedding], 2
    #         )
    #         rnn_outputs, final_state = dynamic_rnn(
    #             LSTMCell(self.hidden_size),
    #             inputs=self.history_embedding,
    #             sequence_length=self.sequence_length,
    #             dtype=tf.float32,
    #             scope="lstm",
    #         )
    #         tf.summary.histogram("LSTM_outputs", rnn_outputs)
    #         return final_state[1]
