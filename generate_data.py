from src.utils.generate_data import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth_map",
                        help="path to depth map", type=str, required=True)
    parser.add_argument("-n", "--normal_map",
                        help="path to normal map", type=str, required=True)
    parser.add_argument("--min_depth_percentile",
                        help="minimum visualization depth percentile",
                        type=float, default=5)
    parser.add_argument("--max_depth_percentile",
                        help="maximum visualization depth percentile",
                        type=float, default=95)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--data_save_folder",
                        help="folder to data save", type=str, required=True)
    parser.add_argument("-n", "--num_generate_img",
                        help="number of generate_img", type=int, default=100)
    args = parser.parse_args()

    print( "folder to data save is ", args.data_save_folder)
    print("number of generate_img is", args.num_generate_img )

    create_dataset(args.data_save_folder, args.num_generate_img)