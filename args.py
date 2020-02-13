from argparse import ArgumentParser
import datetime

def get_args():
    parser = ArgumentParser(description="Multi-label Privacy Policy Classification")
    parser.add_argument('--model_name', type=str, required=False, default='model'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))
    parser.add_argument('--label_set', type=str, required=False, help='Union or Majority', default='Majority')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optim', type=str, required=False, help='SGD or Adam', default='SGD')
    args = parser.parse_args()
    return args
