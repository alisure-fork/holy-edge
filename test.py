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
        test_list = self.tool.read_file_list(file_path)
        test_list = [os.path.join(self.config['download_path'],
                                  self.config['testing']['dir'], img) for img in test_list]
        result_dir = self.config['test_output']

        self.tool.print_info('Writing PNGs at {}'.format(result_dir))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for idx, test_filename in enumerate(test_list):
            im = self._fetch_image(test_filename)
            edge_map = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
            self._save_edge_maps(edge_map, os.path.join(result_dir, os.path.basename(test_filename)))
            self.tool.print_info('Done testing {}, {}'.format(test_filename, im.shape))
            pass

        pass

    def run_1(self, session, test_dir, result_dir):
        if not self.init:
            return

        self.model.setup_testing(session)

        test_list = [os.path.join(test_dir, test_file) for test_file in os.listdir(test_dir)]

        self.tool.print_info('Writing PNGs at {}'.format(result_dir))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for idx, test_filename in enumerate(test_list):
            im = self._fetch_image(test_filename)
            edge_map = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
            self._save_edge_maps(edge_map, os.path.join(result_dir, os.path.basename(test_filename)))
            self.tool.print_info('Done testing {}, {}'.format(test_filename, im.shape))
            pass

        pass

    def run_2(self, session, test_dir, result_dir):
        if not self.init:
            return

        self.model.setup_testing(session)

        self.tool.print_info('Writing PNGs at {}'.format(result_dir))

        for path, dirs, files in os.walk(test_dir):

            result_path = path.replace(test_dir, result_dir)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            for fileName in files:
                if fileName.endswith(".png") or fileName.endswith(".jpg"):
                    now_file_name = os.path.join(path, fileName)
                    im = self._fetch_image(now_file_name)
                    edge_map = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
                    self._save_edge_maps(edge_map, os.path.join(result_path, os.path.basename(fileName)))
                    self.tool.print_info('Done testing {}, {}'.format(fileName, im.shape))
                pass
            pass

        pass

    def _save_edge_maps(self, em_maps, test_output):
        # Take the edge map from the network from side layers and fuse layer
        em_maps = [e[0] for e in em_maps]
        em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]

        for idx, em in enumerate(em_maps):
            em[em < self.config['testing_threshold']] = 0.0
            em = np.tile(255.0 * (1.0 - em), [1, 1, 3])
            em = Image.fromarray(np.uint8(em))
            em.save(test_output)
            pass
        pass

    def _fetch_image(self, test_image):
        image = None
        if os.path.exists(test_image):
            try:
                image = self._capture_pixels(test_image)
            except Exception as err:
                print(self.tool.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err)))
            pass
        return image

    def _capture_pixels(self, image_buffer):
        image = Image.open(image_buffer)
        image = image.resize((self.config['testing']['image_width'], self.config['testing']['image_height']))
        image = np.array(image, np.float32)
        image = self._colorize(image)

        image = image[:, :, self.config['channel_swap']]
        image -= self.config['mean_pixel_value']
        return image

    @staticmethod
    def _colorize(image):
        # BW to 3 channel RGB image
        if image.ndim == 2:
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    pass
