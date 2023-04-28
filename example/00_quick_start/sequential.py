#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import sys
sys.path.append("../../")

# import sys
print(sys.path)
from reco_utils.recommender.deeprec.models.sequential.pamrec import PAMRECModel
from reco_utils.recommender.deeprec.models.sequential.mmoe import MMoEModel_original
from reco_utils.recommender.deeprec.models.sequential.ple import PLEModel
from reco_utils.recommender.deeprec.models.sequential.sharebottom import ShareBottomModel


# sys.path.append("../../reco_utils")
import os

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.config.experimental.enable_op_determinism()
import time

from reco_utils.common.constants import SEED
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams
)
from reco_utils.dataset.sequential_reviews import data_preprocessing#, strong_data_preprocessing
from reco_utils.recommender.deeprec.models.sequential.asvd import A2SVDModel
from reco_utils.recommender.deeprec.models.sequential.caser import CaserModel
from reco_utils.recommender.deeprec.models.sequential.gru4rec import GRU4RecModel
from reco_utils.recommender.deeprec.models.sequential.sasrec import SASRecModel # sasrec_with_time_aware_gumbel
from reco_utils.recommender.deeprec.models.sequential.din import DINModel # din_add_time_feature
from reco_utils.recommender.deeprec.models.sequential.dien import DIENModel
from reco_utils.recommender.deeprec.models.sequential.ncf import NCFModel
from reco_utils.recommender.deeprec.models.sequential.lgn import LGNModel

from reco_utils.recommender.deeprec.io.sequential_iterator import (
    SequentialIterator,
    SASequentialIterator
)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'wechat', 'Dataset name.')
flags.DEFINE_string('eval_metric', 'auc', 'metric to eval')
flags.DEFINE_integer('gpu_id', 0, 'GPU ID.')
flags.DEFINE_integer('val_num_ngs', 0, 'Number of negative instances with a positiver instance for validation.') # 4
flags.DEFINE_integer('test_num_ngs', 0, 'Number of negative instances with a positive instance for testing.') # 99
flags.DEFINE_integer('batch_size', 500, 'Batch size.') # 500
flags.DEFINE_string('save_path', 'ranking', 'Save path.')
flags.DEFINE_string('contrastive_loss', 'triplet', 'Contrastive loss, could be bpr or triplet.')
flags.DEFINE_integer('contrastive_length_threshold', 5, 'Minimum sequence length value to apply contrastive loss.')
flags.DEFINE_integer('contrastive_recent_k', 3, 'Use the most recent k embeddings to compute short-term proxy.')
flags.DEFINE_string('name', 'try', 'Experiment name.')
flags.DEFINE_string('model', 'PAMREC', 'Model name.')# GRU4REC DIN DIEN SASREC SHAREBOTTOM PAMREC PLE Proposed MMOE_ORIGINAL
flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_boolean('write_prediction_to_file', False, 'Whether to write prediction to file.')
flags.DEFINE_boolean('manual_alpha', False, 'Whether to use predefined alpha for long short term fusion.') # ！！！False
flags.DEFINE_float('manual_alpha_value', 1.0, 'Predifined alpha value for long short term fusion.') # !!!
flags.DEFINE_boolean('interest_evolve', True, 'Whether to use a GRU to model interest evolution.')
flags.DEFINE_boolean('predict_long_short', True, 'Predict whether the next interaction is driven by long-term interest or short-term interest.') # !!!! True
flags.DEFINE_integer('is_clip_norm', 1, 'Whether to clip gradient norm.')
flags.DEFINE_string('sequential_model', 'gru', 'Sequential model option, could be gru, lstm, time4lstm.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs.') # 10
flags.DEFINE_integer('early_stop', 5, 'Patience for early stop.') # 5 10 15
# flags.DEFINE_integer('early_stop', 10000, 'Patience for early stop.') # change to step

flags.DEFINE_string('data_path', os.path.join("..", "..", "tests", "resources", "deeprec", "sequential"), 'Data file path.')
flags.DEFINE_integer('train_num_ngs', 0, 'Number of negative instances with a positive instance for training.') # 4
flags.DEFINE_float('sample_rate', 1.0, 'Fraction of samples for training and testing.')
flags.DEFINE_float('embed_l2', 1e-6, 'L2 regulation for embeddings.')
flags.DEFINE_float('layer_l2', 1e-6, 'L2 regulation for layers.')
flags.DEFINE_float('attn_loss_weight', 0.001, 'Loss weight for supervised attention.')
flags.DEFINE_float('triplet_margin', 1.0, 'Margin value for triplet loss.')
flags.DEFINE_float('discrepancy_loss_weight', 0.1, 'Loss weight for discrepancy between long and short term user embedding.') # 0.01 0.2 0.3
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'Loss weight for contrastive of long and short intention.') # 0.1
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('show_step', 500, 'Step for showing metrics.')
flags.DEFINE_integer('bucket_num', 10, 'bucket num for lisanhua')
flags.DEFINE_boolean('add_feature', False, 'add time feature')
flags.DEFINE_float('fuzhu_weight', 0.5, 'add time feature')
flags.DEFINE_integer('eval_step', 2500, 'Step for evaluation.') # 5000
flags.DEFINE_float('noise_train_hist', 0, 'noise_train_hist') # 5000
flags.DEFINE_float('noise_train_listwise', 0, 'noise_train_listwise') # 5000
flags.DEFINE_float('noise_only_predict', 0, 'noise_only_predict') # 5000

def get_model(flags_obj, model_path, summary_path, user_vocab, item_vocab, cate_vocab, train_num_ngs):

    EPOCHS = flags_obj.epochs
    BATCH_SIZE = flags_obj.batch_size
    RANDOM_SEED = 8  # Set None for non-deterministic result
    model_2 = None
    if flags_obj.dataset == 'wechat':
        pairwise_metrics = []
        weighted_metrics = ['wauc', 'wmrr', 'wndcg@2;4;6;8;10', 'whit@2;4;6;8;10']
        max_seq_length = 100
        # time_unit = 's'
    elif flags_obj.dataset == 'takatak':
        pairwise_metrics = []
        weighted_metrics = ['wauc', 'wmrr', 'wndcg@10', 'whit@10', 'wmrr@10']
        max_seq_length = 100
        # time_unit = 's'

    input_creator = SequentialIterator


    if flags_obj.model == 'SHAREBOTTOM':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sharebottom.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2,
                                contrastive_loss=flags_obj.contrastive_loss,
                                triplet_margin=flags_obj.triplet_margin,
                                discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                learning_rate=flags_obj.learning_rate,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                manual_alpha=flags_obj.manual_alpha,
                                manual_alpha_value=flags_obj.manual_alpha_value,
                                interest_evolve=flags_obj.interest_evolve,
                                predict_long_short=flags_obj.predict_long_short,
                                is_clip_norm=flags_obj.is_clip_norm,
                                contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                contrastive_recent_k=flags_obj.contrastive_recent_k,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                sequential_model=flags_obj.sequential_model,
                                # time_unit=time_unit,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = ShareBottomModel(hparams, input_creator, seed=RANDOM_SEED)

    elif flags_obj.model == 'PAMREC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/mmoe.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2,
                                contrastive_loss=flags_obj.contrastive_loss,
                                triplet_margin=flags_obj.triplet_margin,
                                discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                learning_rate=flags_obj.learning_rate,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                manual_alpha=flags_obj.manual_alpha,
                                manual_alpha_value=flags_obj.manual_alpha_value,
                                interest_evolve=flags_obj.interest_evolve,
                                predict_long_short=flags_obj.predict_long_short,
                                is_clip_norm=flags_obj.is_clip_norm,
                                contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                contrastive_recent_k=flags_obj.contrastive_recent_k,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                sequential_model=flags_obj.sequential_model,
                                fuzhu_weight=flags_obj.fuzhu_weight,
                                # time_unit=time_unit,
                                fine_tune=False,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = PAMRECModel(hparams, input_creator, seed=RANDOM_SEED)

    elif flags_obj.model == 'MMOE_ORIGINAL':
        yaml_file = '../../reco_utils/recommender/deeprec/config/mmoe.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                  layer_l2=flags_obj.layer_l2,
                                  contrastive_loss=flags_obj.contrastive_loss,
                                  triplet_margin=flags_obj.triplet_margin,
                                  discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                  contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                  learning_rate=flags_obj.learning_rate,
                                  epochs=EPOCHS,
                                  EARLY_STOP=flags_obj.early_stop,
                                  manual_alpha=flags_obj.manual_alpha,
                                  manual_alpha_value=flags_obj.manual_alpha_value,
                                  interest_evolve=flags_obj.interest_evolve,
                                  predict_long_short=flags_obj.predict_long_short,
                                  is_clip_norm=flags_obj.is_clip_norm,
                                  contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                  contrastive_recent_k=flags_obj.contrastive_recent_k,
                                  batch_size=BATCH_SIZE,
                                  show_step=flags_obj.show_step,
                                  MODEL_DIR=model_path,
                                  SUMMARIES_DIR=summary_path,
                                  user_vocab=user_vocab,
                                  item_vocab=item_vocab,
                                  cate_vocab=cate_vocab,
                                  # need_sample=True,
                                  train_num_ngs=train_num_ngs,
                                  # provides the number of negative instances for each positive instance for loss computation.
                                  max_seq_length=max_seq_length,
                                  pairwise_metrics=pairwise_metrics,
                                  weighted_metrics=weighted_metrics,
                                  sequential_model=flags_obj.sequential_model,
                                  # time_unit=time_unit,
                                  fine_tune=False,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                                  )
        model = MMoEModel_original(hparams, input_creator, seed=RANDOM_SEED)



    elif flags_obj.model == 'PLE':
        yaml_file = '../../reco_utils/recommender/deeprec/config/ple.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2,
                                contrastive_loss=flags_obj.contrastive_loss,
                                triplet_margin=flags_obj.triplet_margin,
                                discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                learning_rate=flags_obj.learning_rate,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                manual_alpha=flags_obj.manual_alpha,
                                manual_alpha_value=flags_obj.manual_alpha_value,
                                interest_evolve=flags_obj.interest_evolve,
                                predict_long_short=flags_obj.predict_long_short,
                                is_clip_norm=flags_obj.is_clip_norm,
                                contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                contrastive_recent_k=flags_obj.contrastive_recent_k,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                sequential_model=flags_obj.sequential_model,
                                # time_unit=time_unit,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = PLEModel(hparams, input_creator, seed=RANDOM_SEED)



    #SasRec
    elif flags_obj.model == 'SASREC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                hidden_size=40,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = SASRecModel(hparams, input_creator, seed=RANDOM_SEED)
    
    #GRU4REC
    elif flags_obj.model == 'GRU4REC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/gru4rec.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                hidden_size=40,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = GRU4RecModel(hparams, input_creator, seed=RANDOM_SEED)

    #DIN
    elif flags_obj.model == 'DIN':
        yaml_file = '../../reco_utils/recommender/deeprec/config/din.yaml'
        print(flags_obj)
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                eval_step=flags_obj.eval_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, hidden_size=40,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,

                    )
        print(hparams)
        model = DINModel(hparams, input_creator, seed=RANDOM_SEED)

    #DIEN
    elif flags_obj.model == 'DIEN':
        yaml_file = '../../reco_utils/recommender/deeprec/config/dien.yaml'
        hparams = prepare_hparams(yaml_file,
                                  dataset=flags_obj.dataset,
                                  bucket_num=flags_obj.bucket_num,
                                  add_feature=flags_obj.add_feature,

                                  embed_l2=flags_obj.embed_l2,
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                # need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                hidden_size=40,
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                  eval_step=flags_obj.eval_step,
                                  noise_train_hist=flags_obj.noise_train_hist,
                                  noise_train_listwise=flags_obj.noise_train_listwise,
                                  noise_only_predict=flags_obj.noise_only_predict,
                    )
        model = DIENModel(hparams, input_creator, seed=RANDOM_SEED)


    return model, model_2


def main(argv):

    flags_obj = FLAGS

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    print('start experiment')

    data_path = os.path.join(flags_obj.data_path, flags_obj.dataset)
    if  flags_obj.dataset == 'wechat':
        reviews_name = 'processed_wechat.csv'
        meta_name = ''

    elif flags_obj.dataset == 'takatak':
        reviews_name = 'processed_takatak.csv'
        meta_name = ''

    # for test
    train_file = os.path.join(data_path, r'train_data')#train_data
    valid_file = os.path.join(data_path, r'valid_data')
    test_file = os.path.join(data_path, r'test_data')
    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
    output_file = os.path.join(data_path, r'output.txt')

    reviews_file = os.path.join(data_path, reviews_name)
    meta_file = os.path.join(data_path, meta_name)
    train_num_ngs = flags_obj.train_num_ngs
    valid_num_ngs = flags_obj.val_num_ngs
    test_num_ngs = flags_obj.test_num_ngs
    sample_rate = flags_obj.sample_rate

    input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]

    if not os.path.exists(train_file):
        data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, dataset=flags_obj.dataset)

    save_path = os.path.join(flags_obj.save_path, flags_obj.model, flags_obj.name)
    model_path = os.path.join(save_path, "model/")
    summary_path = os.path.join(save_path, "summary/")

    model, model_2 = get_model(flags_obj, model_path, summary_path, user_vocab, item_vocab, cate_vocab, train_num_ngs)

    if flags_obj.only_test:
        ckpt_path = tf.train.latest_checkpoint(model_path)
        model.load_model(ckpt_path)
        res = model.run_weighted_eval(test_file, num_ngs=test_num_ngs) # test_num_ngs is the number of negative lines after each positive line in your test_file
        print(res)

        return

    eval_metric = FLAGS.eval_metric

    start_time = time.time()
    # model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric)
    model = model.fit_step(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric)
    # valid_num_ngs is the number of negative lines after each positive line in your valid_file
    # we will evaluate the performance of model on valid_file every epoch
    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))

    ckpt_path = tf.train.latest_checkpoint(model_path)
    print(ckpt_path)
    if model_2 != None:
        model_2.load_model(ckpt_path)
        # model = model_2.fit_step(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric)
        start_time = time.time()
        # model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric)
        model = model_2.fit_step(train_file, valid_file, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric)
        # valid_num_ngs is the number of negative lines after each positive line in your valid_file
        # we will evaluate the performance of model on valid_file every epoch
        end_time = time.time()
        cost_time = end_time - start_time
        print('Time cost for training is {0:.2f} mins'.format((cost_time) / 60.0))
        ckpt_path = tf.train.latest_checkpoint(model_path)
        print(ckpt_path)

    model.load_model(ckpt_path)

    res = model.run_weighted_eval(test_file, num_ngs=test_num_ngs)
    print(flags_obj.name)
    print(res)

    if flags_obj.write_prediction_to_file:
        model = model.predict(test_file, output_file)


if __name__ == "__main__":

    app.run(main)
