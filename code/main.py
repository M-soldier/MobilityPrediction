import argparse

from run.Run import Run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DeepMove",
                        help="model name, options: [Markov, RNN, DeepMove]")
    parser.add_argument("--data", type=str, default="Foursquare", help="dataset type", choices=["Foursquare"])
    parser.add_argument("--root_path", type=str, default="../data/processedData/", help="root path of the data file")
    parser.add_argument("--save_path", type=str, default="../results/")
    parser.add_argument("--loc_emb_size", type=int, default=256, help="location embeddings size")
    parser.add_argument("--tim_emb_size", type=int, default=64, help="time embeddings size")
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--data_name", type=str, default="foursquare")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_step", type=int, default=8)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--L2", type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--epoch_max", type=int, default=20)
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["LSTM", "GRU", "RNN"])
    parser.add_argument("--trace_split", type=str, default="interval")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--loc_size", type=int, default=53262, help="location vocab size")
    parser.add_argument("--tim_size", type=int, default=24*7+1, help="time vocab size")
    parser.add_argument("--uid_size", type=int, default=2252, help="time vocab size")
    parser.add_argument("--uid_emb_size", type=int, default=64, help="time vocab size")
    args = parser.parse_args()

    run = Run(args)
    run.train()
