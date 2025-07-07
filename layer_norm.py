import torch
import torch.nn.functional as F
import torch.mps.profiler as profiler

# import line_profiler as lp
# import memory_profiler as mp



def embed_tokens(token_ids, token_embedding, positional_embedding):
    """Embed tokens using token and positional embeddings.
    Args:
        token_ids: Array of token IDs.
        token_embedding: Token embedding matrix.
        positional_embedding: Positional embedding matrix.
    Returns:
        Embedded tensor of shape (batch_size, sequence_length, num_hidden_dims).
    """
    return token_embedding[token_ids] + positional_embedding.unsqueeze(0)



def layer_norm(input_tensor, gamma_weights, beta_weights, eps=1e-5):
    """Layer normalization function.
    Args:
        x: Input tensor of shape (batch, features).
        gamma: Scale parameter of shape (features,).
        beta: Shift parameter of shape (features,).
        eps: Small constant for numerical stability.
    Returns:
        Normalized tensor of the same shape as x.
    """
    return F.layer_norm(
        input_tensor,
        normalized_shape=input_tensor.shape[-1:],
        weight=gamma_weights,
        bias=beta_weights,
        eps=eps
    )



def relu(input_tensor):
    """Rectified Linear Unit activation function.
    Args:
        input: Input tensor.
    Returns:
        Output tensor with ReLU applied.
    """
    return F.relu(input_tensor, inplace=True)



def softmax(input_tensor, axis=-1):
    """Softmax function.
    Args:
        input_tensor: Input tensor.
        axis: Axis along which to compute softmax.
    Returns:
        Output tensor with softmax applied.
    """
    return F.softmax(input_tensor, dim=axis)



def reshape_for_heads(input_tensor, batch_size, sequence_length, heads, dims_per_head):
    """Reshape input tensor for multi-head attention.
    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, num_hidden_dims).
        batch_size: Batch size.
        sequence_length: Sequence length.
        heads: Number of attention heads.
        dims_per_head: Dimensions per attention head.
    Returns:
        Reshaped tensor of shape (batch_size, heads, sequence_length, dims_per_head).
    """
    return torch.reshape(input_tensor, (
        batch_size, sequence_length, heads, dims_per_head
    )).permute(0, 2, 1, 3)



def causal_mask(size):
    """Create a causal mask for self-attention.
    Args:
        size: Size of the mask (sequence length).
    Returns:
        Causal mask of shape (size, size) with lower triangular part set to True.
    """
    return torch.tril(torch.ones((size, size), dtype=bool))



def self_attention_block(
    input_tensor,
    query_weights,
    key_weights,
    value_weights,
    batch_size,
    sequence_length,
    heads,
    dims_per_head,
    mask,
):
    """
    Self-attention block.
    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, num_hidden_dims).
        query_weights: Weights for query transformation.
        key_weights: Weights for key transformation.
        value_weights: Weights for value transformation.
        batch_size: Batch size.
        sequence_length: Sequence length.
        heads: Number of attention heads.
        dims_per_head: Dimensions per attention head.
        mask: Causal mask for self-attention.
    Returns:
        Output tensor after self-attention.
        Shape: (batch_size, sequence_length, num_hidden_dims).
    """
    query = input_tensor @ query_weights
    key = input_tensor @ key_weights
    value = input_tensor @ value_weights

    query = reshape_for_heads(query, batch_size, sequence_length, heads, dims_per_head)
    key = reshape_for_heads(key, batch_size, sequence_length, heads, dims_per_head)
    value = reshape_for_heads(value, batch_size, sequence_length, heads, dims_per_head)
    # query / key / value shape (batch_size, sequence_length, heads, dims_per_head)
    # key.permute(0, 1, 3, 2) shape (batch_size, sequence_length, dims_per_head, heads)
    # scores shape (batch_size, num_heads, sequence_length, sequence_length)
    # Hence key.permute(0, 1, 3, 2) is used to align dimensions for matrix multiplication.
    scores = torch.divide(query @ torch.permute(key, (0, 1, 3, 2)), (num_hidden_dims ** 0.5))
    # Lower triangular mask to prevent attending to future tokens
    # The token at the ith position can only attend to tokens from 0 to i,
    # since the ith row in the mask contains i True values
    # [:, None, :, :] reshapes the mask to align with the scores tensor, extends in the batch_size dimension
    scores = torch.where(mask[:, None, :, :], scores, -1e9)

    weights = softmax(scores, axis=-1)

    attended = weights @ value
    attended = torch.permute(attended, (0, 2, 1, 3)).reshape(
        batch_size, sequence_length, num_hidden_dims
    )
    return attended



def mlp_block(input_tensor, mlp_weights1, mlp_bias1, mlp_weights2, mlp_bias2):
    """Feedforward MLP block.
    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, num_hidden_dims).
        mlp_weights1: Weights for first layer of MLP.
        mlp_bias1: Bias for first layer of MLP.
        mlp_weights2: Weights for second layer of MLP.
        mlp_bias2: Bias for second layer of MLP.
    Returns:
        Output tensor after MLP.
    """
    hidden = relu(input_tensor @ mlp_weights1.T + mlp_bias1)
    return hidden @ mlp_weights2.T + mlp_bias2



def final_logits(output_tensor, head_weights, gamma_weights, beta_weights):
    """Project output to vocabulary logits after layer norm.
    output_tensor: (batch, seq, hidden)
    head_weights: (vocab_size, hidden)
    Returns: (batch, seq, vocab_size)
    """
    normed_tensor = layer_norm(output_tensor, gamma_weights, beta_weights)
    return normed_tensor @ head_weights.T


def full_block(
    token_ids,
    token_embedding,
    positional_embedding,
    query_weights,
    key_weights,
    value_weights,
    projection_weights,
    projection_bias,
    mlp_weights1,
    mlp_bias1,
    mlp_weights2,
    mlp_bias2,
    gamma_weights1,
    beta_weights1,
    gamma_weights2,
    beta_weights2,
    gamma_weights3,
    beta_weights3,
    batch_size,
    sequence_length,
    heads,
    dims_per_head,
    mask
):
    """Full transformer block with embedding, self-attention, MLP, and final logits.
    Args:
        token_ids: Array of token IDs.
        token_embedding: Token embedding matrix.
        positional_embedding: Positional embedding matrix.
        query_weights: Weights for query transformation.
        key_weights: Weights for key transformation.
        value_weights: Weights for value transformation.
        projection_weights: Weights for projection.
        projection_bias: Bias for projection.
        mlp_weights1: Weights for first layer of MLP.
        mlp_bias1: Bias for first layer of MLP.
        mlp_weights2: Weights for second layer of MLP.
        mlp_bias2: Bias for second layer of MLP.
        gamma_weights1: Scale parameter for first layer normalization.
        beta_weights1: Shift parameter for first layer normalization.
        gamma_weights2: Scale parameter for second layer normalization.
        beta_weights2: Shift parameter for second layer normalization.
        gamma_weights3: Scale parameter for final layer normalization.
        beta_weights3: Shift parameter for final layer normalization.
        batch_size: Batch size.
        sequence_length: Sequence length.
        heads: Number of attention heads.
        dims_per_head: Dimensions per attention head.
        mask: Causal mask for self-attention.
    Returns:
        Output tensor after full transformer block.
    """
    embedded_tokens = embed_tokens(token_ids, token_embedding, positional_embedding)

    layer_normed_input_tensor = layer_norm(
        embedded_tokens, gamma_weights1, beta_weights1
    )

    attention_output = self_attention_block(
        layer_normed_input_tensor,
        query_weights,
        key_weights,
        value_weights,
        batch_size,
        sequence_length,
        heads,
        dims_per_head,
        mask,
    )

    attention_projection = attention_output @ projection_weights + projection_bias
    input_residual = layer_normed_input_tensor + attention_projection

    second_layer_normed_input_tensor = layer_norm(
        input_residual, gamma_weights2, beta_weights2
    )

    mlp_output = mlp_block(
        second_layer_normed_input_tensor,
        mlp_weights1,
        mlp_bias1,
        mlp_weights2,
        mlp_bias2,
    )

    second_input_residual = input_residual + mlp_output

    logits = final_logits(
        second_input_residual, token_embedding, gamma_weights3, beta_weights3
    )
    return softmax(logits, axis=-1)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Starting transformer pass...")
    # Dimensions
    batch_size = 32
    sequence_length = 128
    num_hidden_dims = 768
    heads = 12
    dims_per_head = num_hidden_dims // heads
    dims_mlp = 4 * num_hidden_dims

    vocab_size = 50257  # typical GPT-style vocab


    # The mean (0) and std dev (0.02) match the initialisation range in
    # standard transformer implementations (GPT, BERT, T5)
    # — this range prevents exploding activations and helps training stability.

    # Each token ID and token position maps to a vector in the token embedding matrix.
    token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length), dtype = torch.long).to(device)
    token_embedding = torch.normal(
        0, 0.02, size=(vocab_size, num_hidden_dims)
    ).to(device)
    positional_embedding = torch.normal(
        0, 0.02, size=(sequence_length, num_hidden_dims)
    ).to(device)

    # Learned scale (gamma) and shift (beta) for each feature
    gamma_weights1 = torch.normal(
        1.0, 0.02, size=(num_hidden_dims,)
    ).to(device)
    beta_weights1 = torch.normal(
        0.0, 0.02, size=(num_hidden_dims,)
    ).to(device)
    gamma_weights2 = torch.normal(
        1.0, 0.02, size=(num_hidden_dims,)
    ).to(device)
    beta_weights2 = torch.normal(
        0.0, 0.02, size=(num_hidden_dims,)
    ).to(device)
    gamma_weights3 = torch.normal(
        1.0, 0.02, size=(num_hidden_dims,)
    ).to(device)
    beta_weights3 = torch.normal(
        0.0, 0.02, size=(num_hidden_dims,)
    ).to(device)

    # Projection weights and bias
    projection_weights = torch.randn(num_hidden_dims, num_hidden_dims).to(device)
    projection_bias = torch.randn(num_hidden_dims).to(device)

    # Attention weights
    query_weights = torch.randn(num_hidden_dims, num_hidden_dims).to(device)
    key_weights = torch.randn(num_hidden_dims, num_hidden_dims).to(device)
    value_weights = torch.randn(num_hidden_dims, num_hidden_dims).to(device)

    # MLP weights
    mlp_weights1 = torch.randn(dims_mlp, num_hidden_dims).to(device)
    mlp_bias1 = torch.randn(dims_mlp).to(device)
    mlp_weights2 = torch.randn(num_hidden_dims, dims_mlp).to(device)
    mlp_bias2 = torch.randn(num_hidden_dims).to(device)

    mask = causal_mask(sequence_length).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    profiler.start()
    torch.mps.synchronize()
    full_block_output = full_block(
    token_ids,
    token_embedding,
    positional_embedding,
    query_weights,
    key_weights,
    value_weights,
    projection_weights,
    projection_bias,
    mlp_weights1,
    mlp_bias1,
    mlp_weights2,
    mlp_bias2,
    gamma_weights1,
    beta_weights1,
    gamma_weights2,
    beta_weights2,
    gamma_weights3,
    beta_weights3,
    batch_size,
    sequence_length,
    heads,
    dims_per_head,
    mask
    )
    torch.mps.synchronize()
    profiler.stop()

    print("Tranformer pass completed")
