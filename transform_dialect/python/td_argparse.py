
import argparse
parser = argparse.ArgumentParser(description='Run some matmul examples.')
parser.add_argument('--dump-ptx', nargs='?', type=bool, default=False,
                    help='dump the binary blob with embedded PTX to screen')
parser.add_argument('--dump-full-tensor', nargs='?', type=bool, default=False,
                    help='dump the full restult tensor to screen')
parser.add_argument('--td-repro', nargs='?', type=bool, default=False,
                    help='create a repro for debugging TD schedules')
parser.add_argument('--td-graph-script', nargs='?', type=str, default=None,
                    help='the TD graph-level script to use')
parser.add_argument('--problem-size', nargs='*', type=int, default=None,
                    help='sizes of the problem to run')

def parse_args():
  return parser.parse_args()

