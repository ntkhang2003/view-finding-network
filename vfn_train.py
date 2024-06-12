# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:05:41 2016

@author: jan
"""

import tensorflow as tf
import numpy as np
import time
import network as nw  # Ensure this module is compatible with TF2.x
from tabulate import tabulate
import argparse

def parse_tfrecord_fn(example):
    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, features)
    image = tf.io.decode_raw(parsed_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 6])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return tf.split(image, 2, 2)

def parse_tfrecord_fn_aug(example):
    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, features)
    image = tf.io.decode_raw(parsed_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 6])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.image.random_brightness(image, 0.01)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return tf.split(image, 2, 2)

def get_dataset(filename, batch_size, num_epochs=None, shuffle=False, aug=False):
    dataset = tf.data.TFRecordDataset([filename])
    if aug:
        dataset = dataset.map(parse_tfrecord_fn_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def count_tfrecords(path):
    cnt = 0
    for record in tf.data.TFRecordDataset([path]):
        cnt += 1
    return cnt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default=1000)
    parser.add_argument("--validation_interval", help="Number of iterations after which validation is run", type=int, default=500)
    parser.add_argument("--batch_train", help="Batch size for training", type=int, default=100)
    parser.add_argument("--batch_val", help="Batch size for validation", type=int, default=14)
    parser.add_argument("--checkpoint_interval", help="Number of iterations after which a checkpoint file is written", type=int, default=1000)
    parser.add_argument("--total_steps", help="Number of total training iterations", type=int, default=15000)
    parser.add_argument("--initial_lr", help="Initial learning rate", type=float, default=0.01)
    parser.add_argument("--momentum", help="Momentum coefficient", type=float, default=0.9)
    parser.add_argument("--step_size", help="Number of steps after which the learning rate is reduced", type=int, default=10000)
    parser.add_argument("--step_factor", help="Reduction factor for the learning rate", type=float, default=0.2)
    parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
    parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument("--checkpoint_name", help="Name of the checkpoint files", type=str, default='view_finding_network')
    parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=str2bool, default=True)
    parser.add_argument("--pooling", help="Which pooling function to use", type=str, choices=['max', 'avg'], default='max')
    parser.add_argument("--augment", help="Whether to augment training data or not", type=str2bool, default=True)
    parser.add_argument("--training_db", help="Path to training database", type=str, default='trn.tfrecords')
    parser.add_argument("--validation_db", help="Path to validation database", type=str, default='val.tfrecords')

    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    validation_interval = args.validation_interval
    batch_size_trn = args.batch_train
    batch_size_val = args.batch_val
    checkpoint_interval = args.checkpoint_interval
    total_steps = args.total_steps
    validation_instances = count_tfrecords(args.validation_db)
    initial_lr = args.initial_lr
    momentum_coeff = args.momentum
    step_size = args.step_size
    step_factor = args.step_factor
    parameter_path = args.initial_parameters
    ranking_loss = args.ranking_loss
    experiment_name = args.ranking_loss
    spp = args.spp
    augment_training_data = args.augment

    parameter_table = [["Initial parameters", parameter_path],
                    ["Ranking loss", ranking_loss], ["SPP", spp], ["Pooling", args.pooling],
                    ['Experiment', experiment_name],
                    ['Embedding dim', embedding_dim], ['Batch size', batch_size_trn],
                    ['Initial LR', initial_lr], ['Momentum', momentum_coeff],
                    ['LR Step size', step_size], ['LR Step factor', step_factor],
                    ['Total Steps', total_steps]]

    training_dataset = get_dataset(args.training_db, batch_size_trn, None, True, augment_training_data)
    validation_dataset = get_dataset(args.validation_db, batch_size_val, None, False)

    net_data = np.load(parameter_path, allow_pickle=True).item()
    var_dict = nw.get_variable_dict(net_data)

    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            feature_vec = nw.build_alexconvnet(images, var_dict, embedding_dim, spp, args.pooling)
            loss = nw.loss(feature_vec, nw.build_loss_matrix(batch_size_trn), ranking_loss)
        gradients = tape.gradient(loss, var_dict.values())
        opt.apply_gradients(zip(gradients, var_dict.values()))
        return loss

    @tf.function
    def validation_step(images):
        feature_vec = nw.build_alexconvnet(images, var_dict, embedding_dim, spp, args.pooling)
        loss = nw.loss(feature_vec, nw.build_loss_matrix(batch_size_val), ranking_loss)
        return loss

    opt = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=var_dict)

    current_lr = initial_lr
    validation_history = np.zeros(shape=(total_steps // validation_interval, 3))

    print(tabulate(parameter_table))

    for step in range(total_steps + 1):
        if step % step_size == 0 and step > 0:
            current_lr *= step_factor
            opt.learning_rate.assign(current_lr)
            print("Learning Rate: {}".format(current_lr))
        
        if step % checkpoint_interval == 0:
            checkpoint.save(file_prefix=f'checkpoints/{experiment_name}_step_{step}')

        t0 = time.time()
        for images in training_dataset:
            loss_val = train_step(images)
        t1 = time.time()
        print(f"Iteration {step}: L={loss_val:.4f} dT={t1-t0:.3f}")

        if step % validation_interval == 0 and step > 0:
            val_avg = 0.0
            for val_images in validation_dataset:
                val_loss = validation_step(val_images)
                val_avg += val_loss
            val_avg /= float(validation_instances / batch_size_val)
            validation_history[step // validation_interval - 1] = (step, current_lr, val_avg)
            print(tabulate(validation_history[:step // validation_interval], headers=['Step', 'LR', 'Loss']))
            np.savez(f"{experiment_name}_history.npz", validation=validation_history)

    print(tabulate(parameter_table))
