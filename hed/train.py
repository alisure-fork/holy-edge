import os
import sys
import yaml
import argparse
import tensorflow as tf
from termcolor import colored

from hed.models.vgg16 import Vgg16
from hed.utils.io import IO
from hed.data.data_parser import DataParser


class HEDTrainer():

    def __init__(self, config_file):

        self.io = IO()

        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            print('Error reading config file {}, {}'.format(config_file, err))

    def setup(self):

        try:

            self.model = Vgg16(self.cfgs)
            self.io.print_info('Done initializing VGG-16 model')

        except Exception as err:

            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))

    def run(self, session):

        train_data = DataParser(self.cfgs)

        self.model.setup_training(session)

        opt = tf.train.AdamOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        train = opt.minimize(self.model.loss)

        session.run(tf.global_variables_initializer())

        for idx in range(self.cfgs['max_iterations']):

            im, em, _ = train_data.get_training_batch()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, summary, loss = session.run([train, self.model.merged_summary, self.model.loss],
                                           feed_dict={self.model.images: im, self.model.edgemaps: em},
                                           options=run_options,
                                           run_metadata=run_metadata)

            self.model.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)

            self.io.print_info('[{}/{}] TRAINING loss : {}'.format(idx, self.cfgs['max_iterations'], loss))

            if idx % self.cfgs['save_interval'] == 0:

                saver = tf.train.Saver()
                saver.save(session, os.path.join(self.cfgs['save_dir'], 'models/hed-model'), global_step=idx)

            if idx % self.cfgs['test_interval'] == 0:

                im, em, _ = train_data.get_testing_batch()

                summary, error = session.run([self.model.merged_summary, self.model.error], feed_dict={self.model.images: im, self.model.edgemaps: em})

                self.model.test_writer.add_summary(summary, idx)
                self.io.print_info('[{}/{}] TESTING error : {}'.format(idx, self.cfgs['max_iterations'], error))

        self.model.train_writer.close()