# -*- encoding:utf-8 -*-
import json
import codecs

def load_hyperparam(args):
    with codecs.open(args.config_path, "r", "utf-8") as f:
        param = json.load(f)
    args.emb_size = param.get("emb_size", 768)
    args.hidden_size = param.get("hidden_size", 768)
    args.kernel_size = param.get("kernel_size", 3)
    args.block_size = param.get("block_size", 2)
    args.feedforward_size = param.get("intermediate_size", None)
    args.heads_num = param.get("num_attention_heads", None)
    args.layers_num = param.get("num_hidden_layers", 12)
    args.dropout = param.get("attention_probs_dropout_prob", 0.1)
    
    return args