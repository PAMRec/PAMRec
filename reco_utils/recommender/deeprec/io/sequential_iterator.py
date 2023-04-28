# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import bisect
import copy
import math

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import json
import pickle as pkl
import random
import os
import time

from reco_utils.recommender.deeprec.io.iterator import BaseIterator
from reco_utils.recommender.deeprec.deeprec_utils import load_dict


# __all__ = ["SequentialIterator", "SASequentialIterator", "RecentSASequentialIterator", "ShuffleSASequentialIterator"]
__all__ = ["SequentialIterator"]

bar_border_list = [0.00000000e+00, 4.09545455e-02, 1.28786778e-01, 3.27894289e-01,
              7.66666667e-01, 1.05850847e+00, 1.18596610e+00, 1.53888889e+00,
              2.13000000e+00, 3.18951300e+03]
takatak_bar_border_list_dict = {}
# takatak_bar_border_list_dict[10] = [ 0.        ,  0.09390354,  0.18552465,  0.35144214,  0.59991658,
        # 0.85086416,  1.01774766,  1.06348408,  1.12426334,  1.50696871,
       # 34.54434426] # 10
# takatak_bar_border_list_dict[10] = [0.00000000e+00, 9.16122250e-02, 1.77649819e-01, 3.31436700e-01,
#        5.68412734e-01, 8.27118644e-01, 1.01175994e+00, 1.06276119e+00,
#        1.12528178e+00, 1.52579800e+00, 1.17400791e+02]

takatak_bar_border_list_dict[10] = [0.00000000e+00, 9.42727958e-02, 1.83767085e-01, 3.43377108e-01,
       5.84957075e-01, 8.40688722e-01, 1.01619134e+00, 1.06384801e+00,
       1.12731802e+00, 1.53233455e+00, 1.42410704e+02]

takatak_bar_border_list_dict[8] = [ 0.        ,  0.11222045,  0.25718205,  0.5326864 ,  0.85086416,
        1.03270076,  1.08606422,  1.32310015, 34.54434426]
takatak_bar_border_list_dict[6] = [ 0.        ,  0.14884331,  0.42646411,  0.85086416,  1.0506716 ,
        1.17514919, 34.54434426]
def lisan(x, dataset, num=10):
    if dataset == 'takatak':
        ind = bisect.bisect(takatak_bar_border_list_dict[num], x)
    elif dataset == 'wechat':
        ind = bisect.bisect(bar_border_list, x)
    else:
        raise Exception('the dataset is wrong')
    ind = ind - 1
    if ind < 0:
        return 0
    return ind

class SequentialIterator(BaseIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        # random.seed(8)
        self.noise_train_hist=hparams.noise_train_hist
        self.noise_train_listwise=hparams.noise_train_listwise
        self.noise_only_predict=hparams.noise_only_predict
        print(f"noise_train_hist:{self.noise_train_hist}, noise_train_listwise:{self.noise_train_listwise}, noise_only_predict:{self.noise_only_predict}")
        self.dataset = hparams.dataset
        self.bucket_num = hparams.bucket_num
        dirs, _ = os.path.split(hparams.item_vocab)
        if hparams.dataset == 'wechat':
            meta_path = os.path.join(dirs, "wechat_business_recommenders.csv")
        elif hparams.dataset == 'takatak':
            meta_path = os.path.join(dirs, "takatak_business_recommenders.csv")
        else:
            raise  Exception("there is no the dataset")

        f_meta = open(meta_path, "r")
        self.meta_dict = {}
        for line in f_meta:
            line = line.strip()
            meta_things = line.split("\t")
            iid = int(meta_things[0])
            if iid not in self.meta_dict:
                self.meta_dict[iid] = [int(meta_things[1]), float(meta_things[2])]  # category duration

        instance_path = os.path.join(dirs, "instance_output_1.0")
        columns = ["label_satisfied", "play", "user_id", "item_id", "No", "train_val_test", "cate_id", "duration"]
        #     columns = ["label_satisfied", "label_valid", "user_id", "item_id", "No", "train_val_test" ,"cate_id"]
        import pandas as pd
        # ns_df = pd.read_csv(instance_path, sep="\t", names=columns) # change
        # ns_df = ns_df[ns_df["label_satisfied"] == 1]  # add in 20220511 # change
        # self.items_with_popular = list(set(list(ns_df["item_id"]))) # change

        self.VALID_THRESHOLD = 8
        self.train = False
        self.BEGIN_HISTORY_LEN_MAX = 5
        self.col_spliter = col_spliter
        user_vocab, item_vocab, cate_vocab = (
            hparams.user_vocab,
            hparams.item_vocab,
            hparams.cate_vocab,
        )
        self.userdict, self.itemdict, self.catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )

        self.max_seq_length = hparams.max_seq_length
        self.batch_size = hparams.batch_size
        self.iter_data = dict()

        self.time_unit = hparams.time_unit

        self.graph = graph

        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="labels_satisfied") # change [None, 1]
            self.labels_play = tf.placeholder(tf.float32, [None, 1], name="labels_play") # two label
            self.plays = tf.placeholder(tf.float32, [None, 1], name="plays") # two label
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.items = tf.placeholder(tf.int32, [None], name="items")
            self.cates = tf.placeholder(tf.int32, [None], name="cates")
            self.durations = tf.placeholder(tf.float32, [None], name="durations")
            self.item_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.item_duration_history = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="item_duration_history"
            )
            self.mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.item_satisfied_value_history = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="item_satisfied_value_history"
            )
            self.item_play_value_history = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="item_play_value_history"
            )
            self.item_loop_times_history = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="item_loop_times_history"
            )

            self.satisfied_item_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="satisfied_item_history"
            )
            self.satisfied_cate_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="satisfied_cate_history"
            )
            self.satisfied_duration_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="satisfied_duration_history"
            )
            self.satisfied_play_history = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="satisfied_play_history"
            )
            self.satisfied_mask = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="satisfied_mask"
            )

            # self.time = tf.placeholder(tf.float32, [None], name="time")
            # self.time_diff = tf.placeholder(
            #     tf.float32, [None, self.max_seq_length], name="time_diff"
            # )
            # self.time_from_first_action = tf.placeholder(
            #     tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            # )
            # self.time_to_now = tf.placeholder(
            #     tf.float32, [None, self.max_seq_length], name="time_to_now"
            # )
    def parse_file(self, input_file):
        """Parse the file to a list ready to be used for downstream tasks
        
        Args:
            input_file: One of train, valid or test file which has never been parsed.
        
        Returns: 
            list: A list with parsing result
        """


        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        """Parse one string line into feature values.
            a line was saved as the following format:
            label \t user_hash \t item_hash \t item_cate \t operation_time \t item_history_sequence \t item_cate_history_sequence \t time_history_sequence

        Args:
            line (str): a string indicating one instance

        Returns:
            tuple/list: Parsed results including label, user_id, target_item_id, target_category, item_history, cate_history(, timeinterval_history,
            timelast_history, timenow_history, mid_mask, seq_len, learning_rate)

        """
        if self.train:
            words = line.strip().split(self.col_spliter)
            user_id = self.userdict[words[0]] if words[0] in self.userdict else 0

            item_history_words = words[1].strip().split(",")
            cate_history_words = words[2].strip().split(",")
            duration_history_words = words[3].strip().split(",")
            satisfied_history_words = words[4].strip().split(",")
            play_history_words = words[5].strip().split(",")

            item_history_sequence, cate_history_sequence, satisfied_history_sequence, \
            play_history_sequence, duration_history_sequence = self.get_item_cate_history_sequence(
                item_history_words, cate_history_words, satisfied_history_words,
                play_history_words, duration_history_words, user_id)

            return (
                user_id,
                item_history_sequence,
                cate_history_sequence,
                duration_history_sequence,
                satisfied_history_sequence,
                play_history_sequence
            )
        else:
            words = line.strip().split(self.col_spliter)
            label_satisfied = int(words[0])
            label_play = float(words[1])/1000
            user_id = self.userdict[words[2]] if words[2] in self.userdict else 0
            item_id = self.itemdict[words[3]] if words[3] in self.itemdict else 0
            item_cate = self.catedict[words[4]] if words[4] in self.catedict else 0

            duration = float(words[5]) # 原来是int

            # current_time = float(words[4])

            # time_history_sequence = []

            item_history_words = words[6].strip().split(",")
            cate_history_words = words[7].strip().split(",")
            duration_history_words = words[8].strip().split(",")
            satisfied_history_words = words[9].strip().split(",")
            play_history_words = words[10].strip().split(",")

            item_history_sequence, cate_history_sequence, satisfied_history_sequence,\
            play_history_sequence, duration_history_sequence = self.get_item_cate_history_sequence(
                item_history_words, cate_history_words, satisfied_history_words,
                play_history_words, duration_history_words, user_id)

            return (
                label_satisfied,
                label_play,
                user_id,
                item_id,
                item_cate,
                duration, # add
                item_history_sequence,
                cate_history_sequence,
                duration_history_sequence, # add
                satisfied_history_sequence,
                play_history_sequence
            )

    def get_item_cate_history_sequence(self, item_history_words, cate_history_words, satisfied_history_words,
                                       play_history_words, duration_history_words, user_id):

        item_history_sequence = self.get_item_history_sequence(item_history_words, satisfied_history_words)
        cate_history_sequence = self.get_cate_history_sequence(cate_history_words, satisfied_history_words)
        satisfied_history_sequence = self.get_satisfied_history_sequence(satisfied_history_words)
        valid_history_sequence = self.get_play_history_sequence(play_history_words, duration_history_words)
        duration_history_sequence = self.get_duration_history_sequence(play_history_words, duration_history_words)

        return item_history_sequence, cate_history_sequence, satisfied_history_sequence,\
               valid_history_sequence, duration_history_sequence

    def get_item_history_sequence(self, item_history_words, satisfied_history_words):

        item_history_sequence = []
        for item in item_history_words:
            item_history_sequence.append(
                self.itemdict[item] if item in self.itemdict else 0
            )
        return item_history_sequence

    def get_cate_history_sequence(self, cate_history_words, satisfied_history_words):

        cate_history_sequence = []
        for cate in cate_history_words:
            cate_history_sequence.append(
                self.catedict[cate] if cate in self.catedict else 0
            )
        return cate_history_sequence

    def get_satisfied_history_sequence(self, satisfied_history_words):
        satisfied_history_sequence = [float(word) for word in satisfied_history_words]
        return satisfied_history_sequence

    # def get_duration_history_sequence(self, play_history_words, duration_history_words):
    #     def func(x):
    #         return min(math.floor(x / 0.3), 9)
    #     valid_history_sequence = [func((float(word)/1000)/float(dura)) for word, dura in zip(play_history_words, duration_history_words)]
    #     return valid_history_sequence

    def get_duration_history_sequence(self, play_history_words, duration_history_words):
        # def func(x):
        #     return min(math.floor(x / 0.3), 9)
        valid_history_sequence = [float(dura) for word, dura in zip(play_history_words, duration_history_words)] # 原本是int
        return valid_history_sequence

    def get_play_history_sequence(self, play_history_words, duration_history_words):
        if self.noise_only_predict != 0:
            valid_history_sequence = np.array([max(float(word) / 1000 + np.random.normal(0, self.noise_only_predict), 0) for word, dur in zip(play_history_words, duration_history_words)])
        else:
            valid_history_sequence = np.array([float(word)/1000 for word in play_history_words])
        # valid_history_sequence = np.array([float(word) / 1000 for word in play_history_words])
        return valid_history_sequence

    # def get_duration_history_sequence(self, duration_history_words):
    #     duration_history_sequence = [float(word)for word in duration_history_words]
    #     return duration_history_sequence


    # def get_time_history_sequence(self, time_history_words):
    #
    #     time_history_sequence = [float(i) for i in time_history_words]
    #     return time_history_sequence

    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.
        
        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch. 
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length. 
                Sequences with length lower than min_seq_length will be ignored.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.

        label_satisfied,
        label_play,
        user_id,
        item_id,
        item_cate,
        duration, # add
        item_history_sequence,
        cate_history_sequence,
        duration_history_sequence, # add
        satisfied_history_sequence,
        play_history_sequence

        """

        if os.path.basename(infile) == 'train_data':
            self.train = True
        else:
            self.train = False

        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]

        # if batch_num_ngs > 0:
        #     random.shuffle(lines)

        if self.train is False:
            label_satisfied_list = []
            label_play_list = []
            play_list = []
            user_list = []
            item_list = []
            item_cate_list = []
            duration_list = []
            item_history_batch = []
            item_cate_history_batch = []
            item_duration_history_batch = []
            item_satisfied_value_history_batch = []
            item_play_value_history_batch = []
            item_loop_times_history_batch = []
            cnt = 0
            for line in lines:
                if not line:
                    continue
                (
                    label_satisfied,
                    label_play,
                    user_id,
                    item_id,
                    item_cate,
                    duration,  # add
                    item_history_sequence,
                    cate_history_sequence,
                    duration_history_sequence,  # add
                    satisfied_history_sequence,
                    play_history_sequence
                ) = line
                if len(item_history_sequence) < min_seq_length:
                    continue

                label_satisfied_list.append(label_satisfied)
                label_play_list.append(1.0 if label_play >= 10 else 0.0) # to be continued
                play_list.append(label_play) # to be continued
                user_list.append(user_id)
                item_list.append(item_id)
                item_cate_list.append(item_cate)
                duration_list.append(duration)
                item_history_batch.append(item_history_sequence)
                item_cate_history_batch.append(cate_history_sequence)
                item_duration_history_batch.append(duration_history_sequence)
                item_satisfied_value_history_batch.append(satisfied_history_sequence)
                item_play_value_history_batch.append(play_history_sequence)
                item_loop_times_history_batch.append([lisan(x/y, self.dataset, self.bucket_num) for x, y in zip(play_history_sequence, duration_history_sequence)])
                cnt += 1
                if cnt == self.batch_size:
                    res = self._convert_data(
                        label_satisfied_list,
                        label_play_list,
                        play_list,
                        user_list,
                        item_list,
                        item_cate_list,
                        duration_list,
                        item_history_batch,
                        item_cate_history_batch,
                        item_duration_history_batch,
                        item_satisfied_value_history_batch,
                        item_play_value_history_batch,
                        item_loop_times_history_batch,
                        batch_num_ngs
                    )
                    batch_input = self.gen_feed_dict(res)
                    yield batch_input if batch_input else None
                    label_satisfied_list = []
                    label_play_list = []
                    play_list = []
                    user_list = []
                    item_list = []
                    item_cate_list = []
                    duration_list = []
                    item_history_batch = []
                    item_cate_history_batch = []
                    item_duration_history_batch = []
                    item_satisfied_value_history_batch = []
                    item_play_value_history_batch = []
                    item_loop_times_history_batch = []
                    cnt = 0
            if cnt > 0:
                res = self._convert_data(
                    label_satisfied_list,
                    label_play_list,
                    play_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    duration_list,
                    item_history_batch,
                    item_cate_history_batch,
                    item_duration_history_batch,
                    item_satisfied_value_history_batch,
                    item_play_value_history_batch,
                    item_loop_times_history_batch,
                    batch_num_ngs
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
        else:
            MAX_SEQUENCE = 100
            BEGIN_HISTORY_LEN_MAX = self.BEGIN_HISTORY_LEN_MAX

            def add_a_item_to_hist(item_hist, cate_hist, duration_hist, satisfied_hist,
                                   play_hist, not_satisfied_index_list,
                                   item, cate, duration, satisfied, play):
                '''
                if add successfully, return True, else return False
                '''
                # global item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list
                # VALID_THRESHOLD = 8
                if play < self.VALID_THRESHOLD and satisfied != 1:
                    return False, item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list

                item_hist.append(item)
                cate_hist.append(cate)
                duration_hist.append(duration)
                satisfied_hist.append(satisfied)
                play_hist.append(play if self.noise_train_hist == 0 else max(play + np.random.normal(0, self.noise_train_hist), 0))
                if satisfied == 0:
                    not_satisfied_index_list.append(len(item_hist) - 1)
                if len(item_hist) > MAX_SEQUENCE:
                    if len(not_satisfied_index_list) == 0:
                        # all videos in sequence are satisfied
                        item_hist = item_hist[-MAX_SEQUENCE:]
                        cate_hist = cate_hist[-MAX_SEQUENCE:]
                        duration_hist = duration_hist[-MAX_SEQUENCE:]
                        satisfied_hist = satisfied_hist[-MAX_SEQUENCE:]
                        play_hist = play_hist[-MAX_SEQUENCE:]
                    else:
                        # exist not satisfied video in the sequence. so we can delete video watching ago
                        to_delete_num = len(item_hist) - MAX_SEQUENCE
                        # # temp
                        # if to_delete_num > 1:
                        #     print("there is something wrong")
                        # # temp

                        for j, to_pop_index in enumerate(not_satisfied_index_list[:to_delete_num]):
                            item_hist.pop(to_pop_index - j)
                            cate_hist.pop(to_pop_index - j)
                            duration_hist.pop(to_pop_index - j)
                            satisfied_hist.pop(to_pop_index - j)
                            play_hist.pop(to_pop_index - j)

                        not_satisfied_index_list = not_satisfied_index_list[to_delete_num:]
                        not_satisfied_index_list = [inde - to_delete_num for inde in not_satisfied_index_list]
                return True, item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list

            # build datasource
            data_source = []
            for _, line in enumerate(lines):
                if not line:
                    continue
                (
                    user_id,
                    item_history_sequence,
                    cate_history_sequence,
                    duration_history_sequence,
                    satisfied_history_sequence,
                    play_history_sequence
                ) = line

                satisfied_num = sum(satisfied_history_sequence)
                if satisfied_num < 2:
                    # print(f"there is not enough history for {user_id}")  # to be continued
                    continue
                elif satisfied_num <= BEGIN_HISTORY_LEN_MAX:
                    begin_loc = 1
                else:
                    begin_loc = random.randint(1, BEGIN_HISTORY_LEN_MAX)

                # global item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list
                item_hist = []
                cate_hist = []
                duration_hist = []
                satisfied_hist = []
                play_hist = []
                not_satisfied_index_list = []

                # find the first satisfied item
                i = 0
                while satisfied_history_sequence[i] != 1:
                    _, item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list \
                        = add_a_item_to_hist(item_hist=item_hist,
                                               cate_hist=cate_hist,
                                               duration_hist=duration_hist,
                                               satisfied_hist=satisfied_hist,
                                               play_hist=play_hist,
                                               not_satisfied_index_list=not_satisfied_index_list,
                                               item=item_history_sequence[i],
                                               cate=cate_history_sequence[i],
                                               duration=duration_history_sequence[i],
                                               satisfied=satisfied_history_sequence[i],
                                               play=play_history_sequence[i])
                    i += 1

                # from the first satisfied item, count several hist item
                hist_new_add_count = 0
                while hist_new_add_count < begin_loc:
                    _, item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list \
                        = add_a_item_to_hist(item_hist=item_hist,
                                           cate_hist=cate_hist,
                                           duration_hist=duration_hist,
                                           satisfied_hist=satisfied_hist,
                                           play_hist=play_hist,
                                           not_satisfied_index_list=not_satisfied_index_list,
                                           item=item_history_sequence[i],
                                           cate=cate_history_sequence[i],
                                           duration=duration_history_sequence[i],
                                           satisfied=satisfied_history_sequence[i],
                                           play=play_history_sequence[i])
                    hist_new_add_count += 1
                    i += 1
                # build the record for this user
                a_record = [
                    item_hist,
                    cate_hist,
                    duration_hist,
                    satisfied_hist,
                    play_hist,
                    not_satisfied_index_list,

                    user_id,
                    item_history_sequence[i:],
                    cate_history_sequence[i:],
                    duration_history_sequence[i:],
                    satisfied_history_sequence[i:],
                    play_history_sequence[i:]
                ]
                data_source.append(a_record)

            # base on the data source
            label_satisfied_list = []  # [[1,2,3,4,5], ...]
            label_play_list = []
            play_list = []
            user_list = []
            item_list = []
            item_cate_list = []
            duration_list = []
            item_history_batch = []
            item_cate_history_batch = []
            item_duration_history_batch = []
            item_satisfied_value_history_batch = []
            item_play_value_history_batch = []
            item_loop_times_history_batch = []
            cnt = 0
            random.shuffle(data_source)
            ind_list = [i for i in range(len(data_source))]
            while len(ind_list) > 0:
                new_ind_list = []
                for ind in ind_list:
                    [
                        item_hist,
                        cate_hist,
                        duration_hist,
                        satisfied_hist,
                        play_hist,
                        not_satisfied_index_list,

                        user_id,
                        item_future,
                        cate_future,
                        duration_future,
                        satisfied_future,
                        play_future
                    ] = data_source[ind]
                    future_num = len(item_future)
                    if future_num < BEGIN_HISTORY_LEN_MAX:
                        continue  # to be continued
                    label_satisfied_list.extend(satisfied_future[:BEGIN_HISTORY_LEN_MAX])
                    label_play_list.extend([1.0 if xx >= self.VALID_THRESHOLD else 0.0 for xx in
                                            play_future[:BEGIN_HISTORY_LEN_MAX]])  # to be continued
                    # print(play_future[:BEGIN_HISTORY_LEN_MAX])
                    # play_list.extend([math.log(xx + 0.1)/math.log(900) for xx in # xx / 1000
                    #                   play_future[:BEGIN_HISTORY_LEN_MAX]])


                    # play_list.extend([min(xx/dur, 3)/3 for xx, dur in # xx / 1000
                    #                   zip(play_future[:BEGIN_HISTORY_LEN_MAX], duration_future[:BEGIN_HISTORY_LEN_MAX])])

                    play_list.extend([lisan((xx if self.noise_train_listwise == 0 else max(xx + np.random.normal(0, self.noise_train_listwise), 0)) / dur,
                                            self.dataset, self.bucket_num) for xx, dur in  # xx / 1000
                                      zip(play_future[:BEGIN_HISTORY_LEN_MAX],
                                          duration_future[:BEGIN_HISTORY_LEN_MAX])])

                    # def temp_func(temp_x):
                    #     if temp_x >= 1.1 and temp_x <= 1.9:
                    #         return 0.5
                    #     elif temp_x < 1.1:
                    #         return 0
                    #     else:
                    #         return 1
                    #
                    # play_list.extend([temp_func(xx / dur - 1.5) for xx, dur in  # xx / 1000
                    #                   zip(play_future[:BEGIN_HISTORY_LEN_MAX],
                    #                       duration_future[:BEGIN_HISTORY_LEN_MAX])])

                    user_list.extend([user_id for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_list.extend(item_future[:BEGIN_HISTORY_LEN_MAX])
                    item_cate_list.extend(cate_future[:BEGIN_HISTORY_LEN_MAX])
                    duration_list.extend(duration_future[:BEGIN_HISTORY_LEN_MAX])
                    item_history_batch.extend([copy.deepcopy(item_hist) for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_cate_history_batch.extend([copy.deepcopy(cate_hist) for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_duration_history_batch.extend([copy.deepcopy(duration_hist) for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_satisfied_value_history_batch.extend([copy.deepcopy(satisfied_hist) for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_play_value_history_batch.extend([copy.deepcopy(play_hist) for _ in range(BEGIN_HISTORY_LEN_MAX)])
                    item_loop_times_history_batch.extend([[lisan(x/y, self.dataset, self.bucket_num) for x, y in zip(play_hist, duration_hist)] for _ in range(BEGIN_HISTORY_LEN_MAX)])

                    cnt += BEGIN_HISTORY_LEN_MAX
                    if cnt == self.batch_size:
                        res = self._convert_data(
                            label_satisfied_list,
                            label_play_list,
                            play_list,
                            user_list,
                            item_list,
                            item_cate_list,
                            duration_list,
                            item_history_batch,
                            item_cate_history_batch,
                            item_duration_history_batch,
                            item_satisfied_value_history_batch,
                            item_play_value_history_batch,
                            item_loop_times_history_batch,
                            batch_num_ngs
                        )
                        batch_input = self.gen_feed_dict(res)
                        yield batch_input if batch_input else None
                        label_satisfied_list = []
                        label_play_list = []
                        play_list = []
                        user_list = []
                        item_list = []
                        item_cate_list = []
                        duration_list = []
                        item_history_batch = []
                        item_cate_history_batch = []
                        item_duration_history_batch = []
                        item_satisfied_value_history_batch = []
                        item_play_value_history_batch = []
                        item_loop_times_history_batch =[]
                        cnt = 0

                    if future_num > BEGIN_HISTORY_LEN_MAX:
                        for temp_ind in range(BEGIN_HISTORY_LEN_MAX):
                            _, item_hist, cate_hist, duration_hist, satisfied_hist, play_hist, not_satisfied_index_list = \
                                add_a_item_to_hist(item_hist=item_hist,
                                                   cate_hist=cate_hist,
                                                   duration_hist=duration_hist,
                                                   satisfied_hist=satisfied_hist,
                                                   play_hist=play_hist,
                                                   not_satisfied_index_list=not_satisfied_index_list,
                                                   item=item_future[temp_ind],
                                                   cate=cate_future[temp_ind],
                                                   duration=duration_future[temp_ind],
                                                   satisfied=satisfied_future[temp_ind],
                                                   play=play_future[temp_ind])
                        data_source[ind][0] = item_hist
                        data_source[ind][1] = cate_hist
                        data_source[ind][2] = duration_hist
                        data_source[ind][3] = satisfied_hist
                        data_source[ind][4] = play_hist
                        data_source[ind][5] = not_satisfied_index_list
                        for i in range(7, len(data_source[ind])):
                            data_source[ind][i] = data_source[ind][i][BEGIN_HISTORY_LEN_MAX:]
                        new_ind_list.append(ind)
                ind_list = new_ind_list

            # process the tail
            if cnt > 0:
                res = self._convert_data(
                    label_satisfied_list,
                    label_play_list,
                    play_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    duration_list,
                    item_history_batch,
                    item_cate_history_batch,
                    item_duration_history_batch,
                    item_satisfied_value_history_batch,
                    item_play_value_history_batch,
                    item_loop_times_history_batch,
                    batch_num_ngs
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None

    def _convert_data(
        self,
        label_satisfied_list,
        label_play_list,
        play_list,
        user_list,
        item_list,
        item_cate_list,
        duration_list,
        item_history_batch,
        item_cate_history_batch,
        item_duration_history_batch,
        item_satisfied_value_history_batch,
        item_play_value_history_batch,
        item_loop_times_history_batch,
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_satisfied_list (list): a list of ground-truth labels.
            label_play_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            duration_list: a list of duration.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            item_duration_history_batch (list): a list of duration history
            item_satisfied_value_history_batch (list): a list of satisfied history indexes.
            item_play_value_history_batch (list): a list of play history indexes.
            batch_num_ngs: .

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            # instance_cnt = len(label_satisfied_list)
            # if instance_cnt < 5:
            #     return
            #
            # label_satisfied_list_all = []
            # label_play_list_all = []
            # play_list_all = []
            # item_list_all = []
            # item_cate_list_all = []
            # user_list_all = np.asarray(
            #     [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            # ).flatten()
            # duration_list_all = []
            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            # max_seq_length_batch = self.max_seq_length
            # item_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("int32")
            # item_cate_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("int32")
            # item_duration_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            # item_satisfied_value_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            # item_play_value_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            # item_loop_times_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            #
            #
            #
            # item_satisfied_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("int32")
            # item_satisfied_cate_history_batch_all = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("int32")
            # item_satisfied_mask = np.zeros(
            #     (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            # ).astype("float32")
            # item_satisfied_duration_history_batch_all = np.zeros(
            #     (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            # ).astype("float32")
            #
            #
            # # time_diff_batch = np.zeros(
            # #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # # ).astype("float32")
            # # time_from_first_action_batch = np.zeros(
            # #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # # ).astype("float32")
            # # time_to_now_batch = np.zeros(
            # #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # # ).astype("float32")
            # mask = np.zeros(
            #     (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            # ).astype("float32")
            #
            # for i in range(instance_cnt):
            #     this_length = min(history_lengths[i], max_seq_length_batch)
            #     for index in range(batch_num_ngs + 1):
            #         item_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
            #         item_cate_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(
            #             item_cate_history_batch[i][-this_length:], dtype=np.int32
            #         )
            #         item_duration_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(
            #             item_duration_history_batch[i][-this_length:], dtype=np.float32
            #         )
            #         item_satisfied_value_history_batch_all[
            #         i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(
            #             item_satisfied_value_history_batch[i][-this_length:], dtype=np.float32
            #         )
            #         item_play_value_history_batch_all[
            #         i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(
            #             item_play_value_history_batch[i][-this_length:], dtype=np.float32
            #         )
            #         item_loop_times_history_batch_all[
            #         i * (batch_num_ngs + 1) + index, :this_length
            #         ] = np.asarray(
            #             item_loop_times_history_batch[i][-this_length:], dtype=np.float32
            #         )
            #         mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
            #
            #
            #
            #         satisfied_item_history = []
            #         satisfied_cat_history = []
            #         satisfied_duration_history = []
            #
            #         satisfied_count = 0
            #         for xx in range(this_length):
            #             if item_satisfied_value_history_batch_all[i * (batch_num_ngs + 1) + index][xx] == 1.0:
            #                 satisfied_item_history.append(item_history_batch_all[i * (batch_num_ngs + 1) + index][xx])
            #                 satisfied_cat_history.append(item_cate_history_batch_all[i * (batch_num_ngs + 1) + index][xx])
            #                 satisfied_duration_history.append(item_duration_history_batch_all[i * (batch_num_ngs + 1) + index][xx])
            #                 satisfied_count += 1
            #         item_satisfied_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :satisfied_count
            #         ] = np.asarray(satisfied_item_history, dtype=np.int32)
            #         item_satisfied_cate_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :satisfied_count
            #         ] = np.asarray(
            #             satisfied_cat_history, dtype=np.int32
            #         )
            #         item_satisfied_duration_history_batch_all[
            #             i * (batch_num_ngs + 1) + index, :satisfied_count
            #         ] = np.asarray(
            #             satisfied_duration_history, dtype=np.float32
            #         )
            #         item_satisfied_mask[i * (batch_num_ngs + 1) + index, :satisfied_count] = 1.0
            #
            #         # time_diff_batch[
            #         #     i * (batch_num_ngs + 1) + index, :this_length
            #         # ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
            #         # time_from_first_action_batch[
            #         #     i * (batch_num_ngs + 1) + index, :this_length
            #         # ] = np.asarray(
            #         #     time_from_first_action_list[i][-this_length:], dtype=np.float32
            #         # )
            #         # time_to_now_batch[
            #         #     i * (batch_num_ngs + 1) + index, :this_length
            #         # ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)
            #
            # for i in range(instance_cnt):
            #     positive_item = item_list[i]
            #     label_satisfied_list_all.append(label_satisfied_list[i])
            #     label_play_list_all.append(label_play_list[i])
            #     play_list_all.append(play_list[i])
            #     item_list_all.append(positive_item)
            #     item_cate_list_all.append(item_cate_list[i])
            #     duration_list_all.append(duration_list[i])
            #
            #     count = 0
            #     while batch_num_ngs:
            #         random_value = random.randint(0, instance_cnt - 1)
            #         # random.seed(8)
            #         # random_value = random.randint(0, len(self.items_with_popular) - 1)
            #         negative_item = item_list[random_value]
            #         positive_item_list = item_list[i // self.BEGIN_HISTORY_LEN_MAX: i // self.BEGIN_HISTORY_LEN_MAX + self.BEGIN_HISTORY_LEN_MAX]
            #         # negative_item = self.items_with_popular[random_value]
            #         if negative_item in positive_item_list:
            #             continue
            #         label_satisfied_list_all.append(0)
            #         label_play_list_all.append(0)
            #         play_list_all.append(0)
            #         item_list_all.append(negative_item)
            #         item_cate_list_all.append(item_cate_list[random_value])
            #         duration_list_all.append(duration_list[random_value])
            #         # item_cate_list_all.append(self.meta_dict[negative_item][0]) ############################################
            #         # duration_list_all.append(self.meta_dict[negative_item][1])
            #         count += 1
            #         if count == batch_num_ngs:
            #             break
            #     # if user_list[i] == 4:
            #     #     xxx = [label_satisfied_list_all,
            #     #            label_play_list_all,
            #     #            play_list_all,
            #     #            user_list_all,
            #     #            item_list_all,
            #     #            item_cate_list_all,
            #     #
            #     #            duration_list_all,
            #     #            item_history_batch_all,
            #     #            item_cate_history_batch_all,
            #     #            item_duration_history_batch_all,
            #     #            mask,
            #     #            item_satisfied_value_history_batch_all,
            #     #            item_play_value_history_batch_all]
            #     #     for xxxx in xxx:
            #     #         print(xxxx)
            #     #     print("\n")
            # res = {}
            # res["labels_satisfied"] = np.asarray(label_satisfied_list_all, dtype=np.float32).reshape(-1, 1)
            # res["labels_play"] = np.asarray(label_play_list_all, dtype=np.float32).reshape(-1, 1)
            # res["plays"] = np.asarray(play_list_all, dtype=np.float32).reshape(-1, 1)
            # res["users"] = user_list_all
            # res["items"] = np.asarray(item_list_all, dtype=np.int32)
            # res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            # res["durations"] = np.asarray(duration_list_all, dtype=np.float32)
            #
            # res["item_history"] = item_history_batch_all
            # res["item_cate_history"] = item_cate_history_batch_all
            # res["item_duration_history"] = item_duration_history_batch_all
            # res["mask"] = mask
            #
            # res["item_satisfied_value_history"] = item_satisfied_value_history_batch_all
            # res["item_play_value_history"] = item_play_value_history_batch_all
            #
            # res["satisfied_item_history"] = item_satisfied_history_batch_all
            # res["satisfied_cate_history"] = item_satisfied_cate_history_batch_all
            # res["satisfied_duration_history"] = item_satisfied_duration_history_batch_all
            # res["satisfied_mask"] = item_satisfied_mask
            # return res
            exit(-1)
        else:
            instance_cnt = len(label_satisfied_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_duration_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_satisfied_value_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_play_value_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_loop_times_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")

            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")

            item_satisfied_history_batch_all = np.zeros(
                  (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_satisfied_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")

            item_satisfied_duration_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_satisfied_play_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_satisfied_mask = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")


            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][-this_length:]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[i][-this_length:]
                item_duration_history_batch_all[i, :this_length] = item_duration_history_batch[i][-this_length:]
                item_satisfied_value_history_batch_all[i, :this_length] = item_satisfied_value_history_batch[i][
                    -this_length:
                ]
                item_play_value_history_batch_all[i, :this_length] = item_play_value_history_batch[i][
                    -this_length:
                ]
                item_loop_times_history_batch_all[i, :this_length] = item_loop_times_history_batch[i][
                    -this_length:
                ]

                mask[i, :this_length] = 1.0

                satisfied_item_history = []
                satisfied_cat_history = []
                satisfied_duration_history = []
                # satisfied_play_history = []
                satisfied_looptimes_history = []
                satisfied_count = 0
                for xx in range(this_length):
                    if item_satisfied_value_history_batch_all[i][xx] == 1.0:
                        satisfied_item_history.append(item_history_batch_all[i][xx])
                        satisfied_cat_history.append(item_cate_history_batch_all[i][xx])
                        satisfied_duration_history.append(item_duration_history_batch_all[i][xx])
                        # satisfied_play_history.append(item_play_value_history_batch_all[i][xx])
                        satisfied_looptimes_history.append(lisan(item_play_value_history_batch_all[i][xx]/item_duration_history_batch_all[i][xx], self.dataset, self.bucket_num))
                        satisfied_count += 1
                item_satisfied_history_batch_all[
                    i, :satisfied_count
                ] = np.asarray(satisfied_item_history, dtype=np.int32)

                item_satisfied_cate_history_batch_all[
                    i, :satisfied_count
                ] = np.asarray(
                    satisfied_cat_history, dtype=np.int32
                )
                item_satisfied_duration_history_batch_all[
                    i, :satisfied_count
                ] = np.asarray(
                    satisfied_duration_history, dtype=np.float32
                )
                item_satisfied_play_history_batch_all[
                    i, :satisfied_count
                ] = np.asarray(
                    satisfied_looptimes_history, dtype=np.float32
                )

                item_satisfied_mask[i, :satisfied_count] = 1.0

            # time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                # time_from_first_action_batch[
                #     i, :this_length
                # ] = time_from_first_action_list[i][-this_length:]
                # time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels_satisfied"] = np.asarray(label_satisfied_list, dtype=np.float32).reshape(-1, 1)
            res["labels_play"] = np.asarray(label_play_list, dtype=np.float32).reshape(-1, 1)
            res["plays"] = np.asarray(play_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["durations"] = np.asarray(duration_list, dtype=np.float32)

            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["item_duration_history"] = item_duration_history_batch_all


            res["mask"] = mask

            res["item_satisfied_value_history"] = item_satisfied_value_history_batch_all
            res["item_play_value_history"] = item_play_value_history_batch_all
            res["item_loop_times_history"] = item_loop_times_history_batch_all

            res["satisfied_item_history"] = item_satisfied_history_batch_all
            res["satisfied_cate_history"] = item_satisfied_cate_history_batch_all
            res["satisfied_duration_history"] = item_satisfied_duration_history_batch_all
            res["satisfied_play_history"] = item_satisfied_play_history_batch_all
            res["satisfied_mask"] = item_satisfied_mask

            # res["time"] = np.asarray(time_list, dtype=np.float32)
            # res["time_diff"] = time_diff_batch
            # res["time_from_first_action"] = time_from_first_action_batch
            # res["time_to_now"] = time_to_now_batch
            return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """


        feed_dict = {
            self.labels: data_dict["labels_satisfied"],
            self.labels_play: data_dict["labels_play"],
            self.plays: data_dict["plays"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.durations: data_dict["durations"],
            self.item_history: data_dict["item_history"],
            self.item_cate_history: data_dict["item_cate_history"],
            self.item_duration_history: data_dict["item_duration_history"],
            self.mask: data_dict["mask"],
            self.item_satisfied_value_history: data_dict["item_satisfied_value_history"],
            self.item_play_value_history: data_dict["item_play_value_history"],
            self.item_loop_times_history: data_dict["item_loop_times_history"],
            self.satisfied_item_history: data_dict["satisfied_item_history"],
            self.satisfied_cate_history: data_dict["satisfied_cate_history"],
            self.satisfied_duration_history: data_dict["satisfied_duration_history"],
            self.satisfied_play_history: data_dict["satisfied_play_history"],
            self.satisfied_mask: data_dict["satisfied_mask"],

            # self.time: data_dict["time"],
            # self.time_diff: data_dict["time_diff"],
            # self.time_from_first_action: data_dict["time_from_first_action"],
            # self.time_to_now: data_dict["time_to_now"],
        }
        return feed_dict


class SASequentialIterator(SequentialIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(SASequentialIterator, self).__init__(hparams, graph, col_spliter)
        with self.graph.as_default():
            self.attn_labels_satisfied = tf.placeholder(tf.float32, [None, 1], name="attn_labels_satisfied")

    def _convert_data(
            self,
            label_satisfied_list,
            label_valid_list,
            user_list,
            item_list,
            item_cate_list,
            item_history_batch,
            item_cate_history_batch,
            item_satisfied_value_history_batch,
            item_valid_value_history_batch,
            batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_satisfied_list (list): a list of ground-truth labels.
            label_valid_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            item_satisfied_value_history_batch (list): a list of satisfied history indexes.
            item_valid_value_history_batch (list): a list of valid history indexes.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_satisfied_list)
            if instance_cnt < 5:
                return

            label_satisfied_list_all = []
            label_valid_list_all = []
            attn_label_satisfied_list_all = [] # add
            # attn_label_valid_list_all = [] # add
            item_list_all = []
            item_cate_list_all = []
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            # time_list_all = np.asarray(
            #     [[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32
            # ).flatten()

            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_satisfied_value_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            item_valid_value_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")

            # time_diff_batch = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            # time_from_first_action_batch = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            # time_to_now_batch = np.zeros(
            #     (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            # ).astype("float32")
            mask = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            ).astype("float32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[
                    i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[
                    i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    item_satisfied_value_history_batch_all[
                    i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_satisfied_value_history_batch[i][-this_length:], dtype=np.float32
                    )
                    item_valid_value_history_batch_all[
                    i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_valid_value_history_batch[i][-this_length:], dtype=np.float32
                    )
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    # time_diff_batch[
                    #     i * (batch_num_ngs + 1) + index, :this_length
                    # ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    # time_from_first_action_batch[
                    #     i * (batch_num_ngs + 1) + index, :this_length
                    # ] = np.asarray(
                    #     time_from_first_action_list[i][-this_length:], dtype=np.float32
                    # )
                    # time_to_now_batch[
                    #     i * (batch_num_ngs + 1) + index, :this_length
                    # ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)

            for i in range(instance_cnt):

                this_length = min(history_lengths[i], max_seq_length_batch) # add
                item_cate_history = np.asarray(item_cate_history_batch[i][-this_length:], dtype=np.int32)
                item_satisfied_mask_history = np.asarray(item_satisfied_value_history_batch[i][-this_length:], dtype=np.float32)
                # item_valid_mask_history = np.asarray(item_valid_history_batch[i][-this_length:], dtype=np.float32)

                item_cate_satisfied_history = np.asarray([item_cate_history[i] for i in range(this_length) if item_satisfied_mask_history[i] == 1.0], dtype=np.int32)
                # item_cate_valid_history = np.asarray([item_cate_history[i] for i in range(this_length) if item_valid_mask_history[i] == 1.0], dtype=np.int32)

                positive_item = item_list[i]
                label_satisfied_list_all.append(label_satisfied_list[i])
                label_valid_list_all.append(label_valid_list[i])
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])

                attn_label_satisfied = (item_cate_satisfied_history == item_cate_list[i]).sum() / len(item_cate_satisfied_history) # add
                attn_label_satisfied_list_all.append(attn_label_satisfied)

                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_satisfied_list_all.append(0)
                    label_valid_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])

                    attn_label_satisfied = (item_cate_satisfied_history == item_cate_list[random_value]).sum() / len(
                        item_cate_satisfied_history)  # add
                    attn_label_satisfied_list_all.append(attn_label_satisfied)

                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
            res["labels_satisfied"] = np.asarray(label_satisfied_list_all, dtype=np.float32).reshape(-1, 1)
            res["labels_valid"] = np.asarray(label_valid_list_all, dtype=np.float32).reshape(-1, 1)

            res["attn_labels_satisfied"] = np.asarray(attn_label_satisfied_list_all, dtype=np.float32).reshape(-1, 1) # add

            res["users"] = user_list_all
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask

            res["item_satisfied_value_history"] = item_satisfied_value_history_batch_all
            res["item_valid_value_history"] = item_valid_value_history_batch_all

            # res["time"] = time_list_all
            # res["time_diff"] = time_diff_batch
            # res["time_from_first_action"] = time_from_first_action_batch
            # res["time_to_now"] = time_to_now_batch
            return res

        else:
            instance_cnt = len(label_satisfied_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_satisfied_value_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            item_valid_value_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            # time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
            #     "float32"
            # )
            # time_from_first_action_batch = np.zeros(
            #     (instance_cnt, max_seq_length_batch)
            # ).astype("float32")
            # time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
            #     "float32"
            # )
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")

            attn_label_satisfied_list = [] # add

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][
                                                          -this_length:
                                                          ]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[i][
                                                               -this_length:
                                                               ]

                item_cate_history = np.asarray(item_cate_history_batch[i][-this_length:], dtype=np.int32) # add
                item_satisfied_mask_history = np.asarray(item_satisfied_value_history_batch[i][-this_length:], dtype=np.float32)
                item_cate_satisfied_history = np.asarray(
                    [item_cate_history[xx] for xx in range(this_length) if item_satisfied_mask_history[xx] == 1.0],
                    dtype=np.int32)
                attn_label_satisfied = (item_cate_satisfied_history == item_cate_list[i]).sum() / len(item_cate_satisfied_history)
                attn_label_satisfied_list.append(attn_label_satisfied)


                item_satisfied_value_history_batch_all[i, :this_length] = item_satisfied_value_history_batch[i][
                                                                    -this_length:
                                                                    ]
                item_valid_value_history_batch_all[i, :this_length] = item_valid_value_history_batch[i][
                                                                -this_length:
                                                                ]

                mask[i, :this_length] = 1.0

                # time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                # time_from_first_action_batch[
                #     i, :this_length
                # ] = time_from_first_action_list[i][-this_length:]
                # time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels_satisfied"] = np.asarray(label_satisfied_list, dtype=np.float32).reshape(-1, 1)
            res["attn_labels_satisfied"] = np.asarray(attn_label_satisfied_list, dtype=np.float32).reshape(-1, 1)
            res["labels_valid"] = np.asarray(label_valid_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all

            res["mask"] = mask

            res["item_satisfied_history"] = item_satisfied_value_history_batch_all
            res["item_valid_history"] = item_valid_value_history_batch_all

            # res["time"] = np.asarray(time_list, dtype=np.float32)
            # res["time_diff"] = time_diff_batch
            # res["time_from_first_action"] = time_from_first_action_batch
            # res["time_to_now"] = time_to_now_batch
            return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            # self.time: data_dict["time"],
            # self.time_diff: data_dict["time_diff"],
            # self.time_from_first_action: data_dict["time_from_first_action"],
            # self.time_to_now: data_dict["time_to_now"],
            self.labels: data_dict["labels_satisfied"],
            self.labels_valid: data_dict["labels_valid"],
            self.attn_labels_satisfied: data_dict["attn_labels_satisfied"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.item_history: data_dict["item_history"],
            self.item_cate_history: data_dict["item_cate_history"],
            self.mask: data_dict["mask"],
            self.item_satisfied_value_history: data_dict["item_satisfied_value_history"],
            self.item_valid_value_history: data_dict["item_valid_value_history"],
        }
        return feed_dict


class RecentSASequentialIterator(SASequentialIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(RecentSASequentialIterator, self).__init__(hparams, graph, col_spliter)
        self.counterfactual_recent_k = hparams.counterfactual_recent_k
        self.change_last_k = 0
    def get_item_history_sequence(self, item_history_words, satisfied_history_words):


        item_history_sequence = super(RecentSASequentialIterator, self).get_item_history_sequence(item_history_words, satisfied_history_words)

        # get change_last_k
        if sum(satisfied_history_words) <= self.counterfactual_recent_k:
            self.change_last_k = 0
        else:
            satisfied_index_arr = np.argwhere(np.asarray(satisfied_history_words, dtype=np.int32)==1).reshape(-1)
            satisfied_index_arr.sort()
            self.change_last_k = satisfied_index_arr[satisfied_index_arr.shape[0] - self.counterfactual_recent_k]

        item_history_sequence = item_history_sequence[self.change_last_k:]
        return item_history_sequence

    def get_cate_history_sequence(self, cate_history_words, satisfied_history_words):

        cate_history_sequence = super(RecentSASequentialIterator, self).get_cate_history_sequence(cate_history_words, satisfied_history_words)
        cate_history_sequence = cate_history_sequence[self.change_last_k:]
        return cate_history_sequence

    def get_satisfied_history_sequence(self, satisfied_history_words):
        satisfied_history_sequence = super(RecentSASequentialIterator, self).get_satisfied_history_sequence(satisfied_history_words)
        satisfied_history_sequence = satisfied_history_sequence[self.change_last_k:]
        # satisfied_history_sequence = [float(word) for word in satisfied_history_words]
        return satisfied_history_sequence

    def get_valid_history_sequence(self, valid_history_words, satisfied_history_words):
        valid_history_sequence = super(RecentSASequentialIterator, self).get_valid_history_sequence(
            valid_history_words, satisfied_history_words)
        valid_history_sequence = valid_history_sequence[self.change_last_k:]
        # valid_history_sequence = [float(word) for word in valid_history_words]
        return valid_history_sequence

    # def get_time_history_sequence(self, time_history_words):
    #
    #     time_history_sequence = super(RecentSASequentialIterator, self).get_time_history_sequence(time_history_words)
    #     time_history_sequence = time_history_sequence[-self.counterfactual_recent_k:] if len(time_history_sequence) >= self.counterfactual_recent_k else time_history_sequence
    #     return time_history_sequence


class ShuffleSASequentialIterator(SASequentialIterator):

    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        super(ShuffleSASequentialIterator, self).__init__(hparams, graph, col_spliter)
        self.shuffle_dict = dict()

    def get_item_cate_history_sequence(self, item_history_words, cate_history_words, satisfied_history_words, valid_history_words, user_id):

        item_history_sequence, cate_history_sequence, satisfied_history_sequence, valid_history_sequence = super(ShuffleSASequentialIterator, self).get_item_cate_history_sequence(item_history_words, cate_history_words, satisfied_history_words, valid_history_words, user_id)

        if user_id not in self.shuffle_dict:
            seq_len = len(item_history_sequence)
            order = list(range(seq_len))
            random.shuffle(order)
            self.shuffle_dict[user_id] = order
        order = self.shuffle_dict[user_id]

        item_history_sequence = [item_history_sequence[index] for index in order]
        cate_history_sequence = [cate_history_sequence[index] for index in order]
        satisfied_history_sequence = [satisfied_history_sequence[index] for index in order]
        valid_history_sequence = [valid_history_sequence[index] for index in order]

        return item_history_sequence, cate_history_sequence, satisfied_history_sequence, valid_history_sequence
