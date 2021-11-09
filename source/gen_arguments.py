import argparse
import os


# Turns a dictionary into a class
class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def argparser():
  # for model
  # parser.add_argument(
  #     '--seq_window_lengths',
  #     type=int,
  #     nargs='+',
  #     help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
  # )
  # parser.add_argument(
  #     '--smi_window_lengths',
  #     type=int,
  #     nargs='+',
  #     help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
  # )
  # parser.add_argument(
  #     '--num_windows',
  #     type=int,
  #     nargs='+',
  #     help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
  # )
  # parser.add_argument(
  #     '--num_hidden',
  #     type=int,
  #     default=0,
  #     help='Number of neurons in hidden layer.'
  # )
  # parser.add_argument(
  #     '--num_classes',
  #     type=int,
  #     default=0,
  #     help='Number of classes (families).'
  # )
  # parser.add_argument(
  #     '--max_seq_len',
  #     type=int,
  #     default=0,
  #     help='Length of input sequences.'
  # )
  # parser.add_argument(
  #     '--max_smi_len',
  #     type=int,
  #     default=0,
  #     help='Length of input sequences.'
  # )
  # # for learning
  # parser.add_argument(
  #     '--learning_rate',
  #     type=float,
  #     default=0.001,
  #     help='Initial learning rate.'
  # )
  # parser.add_argument(
  #     '--num_epoch',
  #     type=int,
  #     default=100,
  #     help='Number of epochs to train.'
  # )
  # parser.add_argument(
  #     '--batch_size',
  #     type=int,
  #     default=256,
  #     help='Batch size. Must divide evenly into the dataset sizes.'
  # )
  # parser.add_argument(
  #     '--dataset_path',
  #     type=str,
  #     default='/data/kiba/',
  #     help='Directory for input data.'
  # )
  # parser.add_argument(
  #     '--problem_type',
  #     type=int,
  #     default=1,
  #     help='Type of the prediction problem (1-4)'
  # )
  # parser.add_argument(
  #     '--binary_th',
  #     type=float,
  #     default=0.0,
  #     help='Threshold to split data into binary classes'
  # )
  # parser.add_argument(
  #     '--is_log',
  #     type=int,
  #     default=0,
  #     help='use log transformation for Y'
  # )
  # parser.add_argument(
  #     '--checkpoint_path',
  #     type=str,
  #     default='',
  #     help='Path to write checkpoint file.'
  # )
  # parser.add_argument(
  #     '--log_dir',
  #     type=str,
  #     default='/tmp',
  #     help='Directory for log data.'
  # )



  # FLAGS, unparsed = parser.parse_known_args()

  FLAGS = dict()
  FLAGS['num_windows']= [32]
  FLAGS['seq_window_lengths'] = [8,12] 
  FLAGS['smi_window_lengths'] = [4 ,8] 
  FLAGS['batch_size']= 256 
  FLAGS['num_epoch'] = 100 
  FLAGS['max_seq_len'] = 1000 
  FLAGS['max_smi_len'] = 100 
  FLAGS['dataset_path'] = '../data/kiba/'
  FLAGS['problem_type'] = 1 
  FLAGS['is_log'] = 0 
  FLAGS['log_dir'] ='logs/'
  FLAGS = Dict2Class(FLAGS)
  # check validity
  #assert( len(FLAGS.window_lengths) == len(FLAGS.num_windows) )

  return FLAGS




def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  #print(msg)

