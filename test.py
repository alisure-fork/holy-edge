import os
import numpy as np
from PIL import Image
from tool import Tool
import tensorflow as tf
from vgg16 import Vgg16


class HEDTester(object):

    def __init__(self, config_file):
        self.tool = Tool()
        self.init = True
        self.config = self.tool.read_yaml_file(config_file)
        pass

    def setup(self, session):
        try:
            self.model = Vgg16(self.config, run='testing')
            meta_model_file = os.path.join(self.config['models_dir'],
                                           'hed-model-{}'.format(self.config['test_snapshot']))
            tf.train.Saver().restore(session, meta_model_file)
            self.tool.print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))
        except Exception as err:
            self.tool.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False
            pass
        pass

    def run(self, session):
        if not self.init:
            return

        self.model.setup_testing(session)

        file_path = os.path.join(self.config['download_path'], self.config['testing']['list'])
        train_list = self.tool.read_file_list(file_path)

        self.tool.print_info('Writing PNGs at {}'.format(self.config['test_output']))

        for idx, img in enumerate(train_list):
            test_filename = os.path.join(self.config['download_path'], self.config['testing']['dir'], img)
            im = self.fetch_image(test_filename)
            edge_map = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
            self.save_edge_maps(edge_map, idx)
            self.tool.print_info('Done testing {}, {}'.format(test_filename, im.shape))
            pass

        pass

    def save_edge_maps(self, em_maps, index):
        # Take the edge map from the network from side layers and fuse layer
        em_maps = [e[0] for e in em_maps]
        em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]

        for idx, em in enumerate(em_maps):
            em[em < self.config['testing_threshold']] = 0.0
            em = np.tile(255.0 * (1.0 - em), [1, 1, 3])
            em = Image.fromarray(np.uint8(em))
            if not os.path.exists(self.config['test_output']):
                os.makedirs(self.config['test_output'])
            em.save(os.path.join(self.config['test_output'], 'testing-{}-{:03}.png'.format(index, idx)))
            pass
        pass

    def fetch_image(self, test_image):
        image = None
        if os.path.exists(test_image):
            try:
                image = self.capture_pixels(test_image)
            except Exception as err:
                print(self.tool.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err)))
            pass
        return image

    def capture_pixels(self, image_buffer):
        image = Image.open(image_buffer)
        image = image.resize((self.config['testing']['image_width'], self.config['testing']['image_height']))
        image = np.array(image, np.float32)
        image = self.colorize(image)

        image = image[:, :, self.config['channel_swap']]
        image -= self.config['mean_pixel_value']
        return image

    @staticmethod
    def colorize(image):
        # BW to 3 channel RGB image
        if image.ndim == 2:
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    pass
