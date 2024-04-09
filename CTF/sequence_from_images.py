import os
import sys
from sequence_grabber import SequenceRunner
from 

def io_organize(path):
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = os.path.join(args.output_folder, timestr + '_iris-mm-' + args.iris_mm)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

def main():
    folder = "C:\\Users\lab1\projects\Microscope_production_tests\CTF\log\\2024-03-20_09-24-59_iris-mm-12"
    
    
if __name__ == '__main__':
    # Initialize Args instance
    args = Args()
    ask = True
