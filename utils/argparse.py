import argparse

def get_args():
    parser = argparse.ArgumentParser(description="get the parameter")

    parser.add_argument('--numClient', type=int, default=10)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--model', type=str, default='ResNet', choices=['CNN', 'ResNet'])
    parser.add_argument('--path', default='./')
    
    parser.add_argument('--globalEpochs', type=int, default=10)
    parser.add_argument('--localEpochs', type=int, default=1)


    parser.add_argument('--optimizet', default='sgd')
    parser.add_argument('--batchsize',type=int, default=32)
    parser.add_argument('--lr',type=float, default=0.005)
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay for optim')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    #datasplit
    parser.add_argument("--byclass", default=False, action='store_true')
    parser.add_argument("--iid", default=True,  action='store_false', help="if use in termal, means noniid")
    parser.add_argument('--alpha', type=float, default=0.0)

    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--gpuIndex', type=int, default=0)

    parser.add_argument("--backdoor", default=False,  action='store_true')
    parser.add_argument("--backRate", type=float, default=0.5)

    #unlearning
    parser.add_argument('--unlearningChoice', action='store_true', default=False)
    parser.add_argument('--unlearningEpochs', type=int, default=20)
    parser.add_argument('--unlearningMethods', type=str, default="cont", choices=["continuous", "gradAsc", "active", "retrain", "cont", "cont_w", "rapid", "ifu", "ffmu", "can"])

    parser.add_argument("--surcon", default=False,  action='store_true')

    parser.add_argument("--tau", type=float, default=0.0007, help="temperature")
    parser.add_argument("--mu", type=float, default=0.0001)

    parser.add_argument("--embeddingLen", type=int, default=128)

    # Gen
    parser.add_argument('-nd', "--noise_dim", type=int, default=256)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=256)
    parser.add_argument('-te', "--trainEpochs", type=int, default=20)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)

    # Fine-tune
    parser.add_argument('-flr', "--fineTuneLearningRate", type=float, default=0.003)
    parser.add_argument('-fe', "--fineTuneEpochs", type=int, default=20)
    parser.add_argument('-ge', "--gamme", type=float, default=0.001)

    args=parser.parse_args()
    return args
