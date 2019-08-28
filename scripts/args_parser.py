import argparse


def getArgs(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        required=True, help="Data home")
    parser.add_argument("--k", type=int, default=100,
                        help="Top-k for reranking")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level. Defaults to INFO")
    parser.add_argument("--anserini_path", type=str,
                        default="/ssd2/arthur/Anserini/anserini.jar", help="anserini.jar path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force overwrite of everything")
    parser.add_argument("--run_retrieval", action="store_true",
                        help="Should the retrieval step run?")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Train file path to be used")
    parser.add_argument("--dev_file", type=str, required=True,
                        help="Development file path to be used")
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bert/XLNet Model to be used")
    parser.add_argument("--train_batch_size", type=int,
                        default=32, help="Size of the training batch")
    parser.add_argument("--dev_batch_size", type=int,
                        default=128, help="Size of the dev batch")
    parser.add_argument("--n_epochs", type=int,
                        default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-5, help="Learning rate for fine-tunning BERT")
    parser.add_argument("--limit_gpus:", type=int, default=-1,
                        help="Limit number of GPUs to be used. Set to -1 to use all")
    parser.add_argument("--per_gpu_train_batch_size", type=int,
                        default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=0)
    return parser.parse_args(argv)
