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
    parser.add_argument("--bert-model", type=str, required=True,
                        help="Bert/XLNet Model to be used")
    return parser.parse_args(argv)
