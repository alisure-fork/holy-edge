import os
from tool import Tool
from vgg16 import Vgg16
import tensorflow as tf
from data import DataParser


class HEDTrainer(object):

    def __init__(self, config_file):
        self.tool = Tool()
        self.init = True
        self.config = self.tool.read_yaml_file(config_file)
        pass

    def setup(self):
        try:
            self.model = Vgg16(self.config)
            self.tool.print_info('Done initializing VGG-16 model')

            dirs = ['train', 'val', 'test']
            dirs = [os.path.join(self.config['train_dir'] + '/{}'.format(d)) for d in dirs]
            dirs.append(os.path.join(self.config['models_dir']))
            _ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]
        except Exception as err:
            self.tool.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False
            pass
        pass

    def run(self, session):
        if not self.init:
            return

        train_data = DataParser(self.config)
        self.model.setup_training(session)

        opt = tf.train.AdamOptimizer(self.config['optimizer_params']['learning_rate'])
        train = opt.minimize(self.model.loss)

        session.run(tf.global_variables_initializer())
        for idx in range(self.config['max_iterations']):
            im, em, _ = train_data.get_training_batch()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, summary, loss = session.run([train, self.model.merged_summary, self.model.loss],
                                           feed_dict={self.model.images: im, self.model.edge_maps: em},
                                           options=run_options,
                                           run_metadata=run_metadata)

            self.model.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)

            self.tool.print_info('[{}/{}] TRAINING loss : {}'.format(idx, self.config['max_iterations'], loss))
            if idx % self.config['save_interval'] == 0:
                saver = tf.train.Saver()
                saver.save(session, os.path.join(self.config['models_dir'], 'hed-model'), global_step=idx)
                pass

            if idx % self.config['val_interval'] == 0:
                im, em, _ = train_data.get_validation_batch()
                summary, error = session.run([self.model.merged_summary, self.model.error],
                                             feed_dict={self.model.images: im, self.model.edge_maps: em})
                self.model.val_writer.add_summary(summary, idx)
                self.tool.print_info('[{}/{}] VALIDATION error : {}'.format(idx, self.config['max_iterations'], error))
                pass
            pass

        self.model.train_writer.close()

        pass

    pass
