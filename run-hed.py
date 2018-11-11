import argparse
from tool import Tool
import tensorflow as tf
from test import HEDTester
from train import HEDTrainer


def main(args):

    if not (args.run_train or args.run_test or args.download_data):
        print('Set at least one of the options --train | --test | --download-data')
        parser.print_help()
        return

    if args.download_data:
        tool = Tool()
        config = tool.read_yaml_file(args.config_file)
        tool.download_data(config['rar_file'], config['download_path'])
        pass

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    if args.run_train:
        trainer = HEDTrainer(args.config_file)
        trainer.setup()
        trainer.run(sess)
        pass

    if args.run_test:
        tester = HEDTester(args.config_file)
        tester.setup(sess)
        tester.run(sess)
        pass

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training/Testing DL models(Concepts/Captions)')
    parser.add_argument('--config-file', dest='config_file', type=str, default="./configs/hed.yaml", help='config')
    parser.add_argument('--train', dest='run_train', action='store_true', default=False, help='Launch training')
    parser.add_argument('--test', dest='run_test', action='store_true', default=True, help='testing')
    parser.add_argument('--download-data', dest='download_data', action='store_true', default=False, help='Download')

    main(parser.parse_args())
