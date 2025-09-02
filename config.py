def def_config():
    return {
        "lr":1e-3,
        "n_emb":512,
        "block_size":18, 
        "batch_size":32,
        "num_hidden":512,
        "num_head":4,
        "dropout":0.1,
        "max_tokens":1000,
        "epochs":5000
    }