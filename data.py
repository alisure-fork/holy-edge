import os
import numpy as np
from PIL import Image
from tool import Tool


class DataParser(object):

    def __init__(self, config):
        self.tool = Tool()
        self.config = config

        self.train_file = os.path.join(config['download_path'], config['training']['list'])
        self.train_data_dir = os.path.join(config['download_path'], config['training']['dir'])
        self.training_pairs = self.tool.read_file_list(self.train_file)

        self.samples = self.tool.split_pair_names(self.training_pairs, self.train_data_dir)
        self.tool.print_info('Training data set-up from {}'.format(os.path.join(self.train_file)))
        self.n_samples = len(self.training_pairs)

        self.all_ids = list(range(self.n_samples))
        np.random.shuffle(self.all_ids)

        self.training_ids = self.all_ids[:int(self.config['train_split'] * len(self.training_pairs))]
        self.validation_ids = self.all_ids[int(self.config['train_split'] * len(self.training_pairs)):]

        self.tool.print_info('Training samples {}'.format(len(self.training_ids)))
        self.tool.print_info('Validation samples {}'.format(len(self.validation_ids)))
        pass

    def get_training_batch(self):
        batch_ids = np.random.choice(self.training_ids, self.config['batch_size_train'])
        return self.get_batch(batch_ids)

    def get_validation_batch(self):
        batch_ids = np.random.choice(self.validation_ids, self.config['batch_size_val'])
        return self.get_batch(batch_ids)

    def get_batch(self, batch):

        file_names = []
        images = []
        edge_maps = []

        for idx, b in enumerate(batch):
            im = Image.open(self.samples[b][0])
            em = Image.open(self.samples[b][1])

            im = im.resize((self.config['training']['image_width'], self.config['training']['image_height']))
            em = em.resize((self.config['training']['image_width'], self.config['training']['image_height']))

            im = np.array(im, dtype=np.float32)
            im = im[:, :, self.config['channel_swap']]
            im -= self.config['mean_pixel_value']
            em = np.array(em.convert('L'), dtype=np.float32)

            if self.config['target_regression']:
                bin_em = em / 255.0
            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1
                pass

            # Some edge maps have 3 channels some do not
            bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
            # To fit [batch_size, H, W, 1] output of the network
            bin_em = np.expand_dims(bin_em, 2)

            images.append(im)
            edge_maps.append(bin_em)
            file_names.append(self.samples[b])
            pass

        return images, edge_maps, file_names

    pass
