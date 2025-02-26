import argparse

def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser_make() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=232)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--train_data_path", type=str, default="data/train.csv")
    parser.add_argument("--test_data_path", type=str, default="data/test.csv")
    parser.add_argument("--question_embedding_path", type=str, default="data/question_embeddings.pth")

    parser.add_argument("--embedding_save_path", type=str, default="data/model_embeddings.pth")
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--model_load_path", type=str, default=None)
    parser.add_argument("--output_save_path",type = str ,default = None)

    parser.add_argument("--model_use_train_l",type=int,default=0)
    parser.add_argument("--model_use_train_r",type=int,default=112)
    parser.add_argument("--model_use_test_l",type=int,default=0)
    parser.add_argument("--model_use_test_r",type=int,default=112)

    parser.add_argument("--is_dyn",type=str2bool,default=False)
    parser.add_argument("--frozen",type=str2bool,default=False)
    return parser 