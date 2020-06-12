import numpy as np
import torch


def create_embedding_layer(weight_matrix: np.ndarray, non_trainable=True):
    weight_matrix = torch.from_numpy(weight_matrix)
    num_embeddings, embedding_dim = weight_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weight_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
