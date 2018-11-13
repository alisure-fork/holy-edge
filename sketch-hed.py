import argparse
from tool import Tool
import tensorflow as tf
from test import HEDTester
from train import HEDTrainer


if __name__ == '__main__':

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    tester = HEDTester(config_file="./configs/hed.yaml")
    tester.setup(sess)
    tester.run_2(sess, test_dir="/home/ubuntu/data1.5TB/Sketchy/rendered_256x256/256x256/photo/tx_000100000000",
                 result_dir="/home/ubuntu/data1.5TB/Sketchy/edge-hed")

    pass
