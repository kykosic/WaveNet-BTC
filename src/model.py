"""
    Model class for Multi-WaveNet
"""
from __future__ import division
import os
import random
import string
import logging
import numpy as np
import tensorflow as tf

from src.layers import dilated_conv, conv1d


class WaveNet(object):
    """ Wrapper for the multi-WaveNet architecture """

    def __init__(self, time_steps, columns, num_filters, num_layers, learning_rate,
                 regularization, num_iters, log_dir, seed=None):
        self.time_steps = time_steps
        self.columns = columns
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_iters = num_iters
        self.log_dir = log_dir
        self.seed = seed

        assert self.num_layers >= 2, "Must use at least 2 dilation layers"

        self._build_graph()
        
    def _build_graph(self):
        """ Build the TensorFlow Graph """
        tf.reset_default_graph()

        self.inputs = dict()
        self.targets = dict()

        with tf.variable_scope('input'):
            for f in self.columns:
                self.inputs[f] = tf.placeholder(tf.float32, (None, self.time_steps), 'input_%s' % f)
                self.targets[f] = tf.placeholder(tf.float32, (None, self.time_steps), 'target_%s' % f)
        
        # Create wavenet for each column being regressed
        self.costs = dict()
        self.optimizers = dict()
        self.outputs = dict()
        for column in self.columns:
            with tf.variable_scope(column):

                # Input layer with conditioning gates
                conditions = list()
                with tf.variable_scope('input_layer'):
                    for k in self.inputs.keys():
                        with tf.variable_scope('condition_%s' % k):
                            dilation = 1
                            X = tf.expand_dims(self.inputs[k], 2)
                            h = dilated_conv(X, self.num_filters, name='input_conv_%s' % k, seed=self.seed)
                            skip = conv1d(X, self.num_filters, filter_width=1, name='skip_%s' % k, 
                                    activation=None, seed=self.seed)
                            conditions.append(h + skip)

                    output = tf.add_n(conditions)

                # Intermediate dilation layers
                with tf.variable_scope('dilated_stack'):
                    for i in range(self.num_layers - 1):
                        with tf.variable_scope('layer_%d' % i):
                            dilation = 2 ** (i + 1)
                            h = dilated_conv(output, self.num_filters, dilation=dilation, name='dilated_conv', 
                                    seed=self.seed)
                            output = h + output

                # Output layer
                with tf.variable_scope('output_layer'):
                    output = conv1d(output, 1, filter_width=1, name='output_conv', activation=None,
                            seed=self.seed)
                    self.outputs[column] = tf.squeeze(output, [2])

            # Optimization
            with tf.variable_scope('optimize_%s' % column):
                mae_cost = tf.reduce_mean(tf.losses.absolute_difference(
                    labels=self.targets[column], predictions=self.outputs[column]))
                trainable = tf.trainable_variables(scope=column)
                l2_cost = tf.add_n([tf.nn.l2_loss(v) for v in trainable if not ('bias' in v.name)])
                self.costs[column] = mae_cost + self.regularization / 2 * l2_cost
                tf.summary.scalar('loss_%s' % column, self.costs[column])

                self.optimizers[column] = tf.train.AdamOptimizer(self.learning_rate).minimize(self.costs[column])

        # Tensorboard output
        run_id = ''.join(random.choice(string.uppercase) for x in range(6))
        self.run_dir = os.path.join(self.log_dir, run_id)
        self.writer = tf.summary.FileWriter(self.run_dir)
        self.writer.add_graph(tf.get_default_graph())
        self.run_metadata = tf.RunMetadata()
        self.summaries = tf.summary.merge_all()

        logging.info("Graph for run %s created", run_id)

    def __enter__(self):
        """ Create tf.Session """
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self, *args):
        """ Stop tf.Session """
        self.sess.close()

    def train(self, targets, features):
        """ Train the model, logging to TensorBoard """
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
        checkpoint_path = os.path.join(self.run_dir, 'model.ckpt')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        logging.info("Writing TensorBoard log to %s", self.run_dir)

        # Sort input dictionaries into the feed dictionary
        feed_dict = dict()
        for column in self.columns:
            feed_dict[self.inputs[column]] = features[column]
            feed_dict[self.targets[column]] = targets[column]

        for step in range(self.num_iters):
            opts = [self.optimizers[f] for f in self.columns]
            _ = self.sess.run(opts, feed_dict=feed_dict)

            # Save summaries every 100 steps
            if (step % 100) == 0:
                summary = self.sess.run([self.summaries], feed_dict=feed_dict)[0]
                self.writer.add_summary(summary, step)
                self.writer.flush()

            # Print cost to console every 1000 steps, also store metadata
            if (step % 1000) == 0:
                costs = [self.costs[f] for f in self.columns]
                costs = self.sess.run(costs, feed_dict=feed_dict, 
                        run_metadata=self.run_metadata, options=run_options)
                self.writer.add_run_metadata(self.run_metadata, 'step_%d' % step)

                cost = ", ".join(map(lambda x: "%.06f" % x, costs))
                logging.info("Losses at step %d: %s", step, cost)

        costs = [self.costs[f] for f in self.columns]
        costs = self.sess.run(costs, feed_dict=feed_dict)
        cost = ", ".join(map(lambda x: "%.06f" % x, costs))
        logging.info("Final loss: %s", cost)

        # Save final checkpoint of model
        logging.info("Storing model checkpoint %s", checkpoint_path)
        saver.save(self.sess, checkpoint_path, global_step=step)

        # Format output back into dictionary form
        outputs = [self.outputs[f] for f in self.columns]
        outputs = self.sess.run(outputs, feed_dict=feed_dict)

        out_dict = dict()
        for i, f in enumerate(self.columns):
            out_dict[f] = outputs[i]

        return out_dict
        
    def generate(self, num_steps, features):
        """ Dynamically forcast the future for num_steps """
        forecast = dict()
        for f in self.columns:
            forecast[f] = list()

        for step in range(num_steps):

            feed_dict = dict()
            for f in self.columns:
                feed_dict[self.inputs[f]] = features[f]

            outputs = [self.outputs[f] for f in self.columns]
            outputs = self.sess.run(outputs, feed_dict=feed_dict)

            for i, f in enumerate(self.columns):
                features[f][0, :] = np.append(features[f][0, 1:], outputs[i][0, -1])
                forecast[f].append(outputs[i][0, -1])
        
        for f in self.columns:
            forecast[f] = np.array(forecast[f]).reshape(1, -1)

        return forecast
    