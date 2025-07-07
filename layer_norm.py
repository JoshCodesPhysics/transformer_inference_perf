import numpy as np

# import line_profiler as lp
# import memory_profiler as mp


@profile
def embed_tokens(token_ids, token_embedding, positional_embedding):
    """Embed tokens using token and positional embeddings.
    Args:
        token_ids: Array of token IDs.
        token_embedding: Token embedding matrix.
        positional_embedding: Positional embedding matrix.
    Returns:
        Embedded tensor of shape (batch_size, sequence_length, num_hidden_dims).
    """
    embedded_tokens = token_embedding[token_ids]
    return embedded_tokens + positional_embedding[np.newaxis, :, :]


@profile
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
    mean = np.mean(input_tensor, axis=-1, keepdims=True)
    var = np.var(input_tensor, axis=-1, keepdims=True)
    # Eps prevents division by zero
    norm = (input_tensor - mean) / np.sqrt(var + eps)
    return norm * gamma_weights + beta_weights


@profile
def relu(input_tensor):
    """Rectified Linear Unit activation function.
    Args:
        input: Input tensor.
    Returns:
        Output tensor with ReLU applied.
    """
    return np.maximum(0, input_tensor)


@profile
def softmax(input_tensor, axis=-1):
    """Softmax function.
    Args:
        input_tensor: Input tensor.
        axis: Axis along which to compute softmax.
    Returns:
        Output tensor with softmax applied.
    """
    input_tensor = input_tensor - np.max(
        input_tensor, axis=axis, keepdims=True
    )  # prevent overflow
    exp = np.exp(input_tensor)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@profile
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
    if input_tensor.ndim != 3:
        raise ValueError(
            f"Expected input tensor to have 3 dimensions, got {input_tensor.ndim} dimensions."
        )
    return input_tensor.reshape(
        batch_size, sequence_length, heads, dims_per_head
    ).transpose(0, 2, 1, 3)


@profile
def causal_mask(size):
    """Create a causal mask for self-attention.
    Args:
        size: Size of the mask (sequence length).
    Returns:
        Causal mask of shape (size, size) with lower triangular part set to True.
    """
    return np.tril(np.ones((size, size), dtype=bool))


def dropout(input_tensor, dropout_rate):
    """Apply dropout to input tensor.
    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, num_hidden_dims).
        dropout_rate: Dropout rate (0.0 to 1.0).
    Returns:
        Output tensor with dropout applied.
    """
    if dropout_rate <= 0.0:
        return input_tensor
    mask = np.random.binomial(1, 1 - dropout_rate, size=input_tensor.shape).astype(
        input_tensor.dtype
    )
    return input_tensor * mask / (1 - dropout_rate)


@profile
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
    # key.transpose(0, 1, 3, 2) shape (batch_size, sequence_length, dims_per_head, heads)
    # scores shape (batch_size, num_heads, sequence_length, sequence_length)
    # Hence key.transpose(0, 1, 3, 2) is used to align dimensions for matrix multiplication.
    scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(num_hidden_dims)
    # Lower triangular mask to prevent attending to future tokens
    # The token at the ith position can only attend to tokens from 0 to i,
    # since the ith row in the mask contains i True values
    # [:, None, :, :] reshapes the mask to align with the scores tensor, extends in the batch_size dimension
    scores = np.where(mask[:, None, :, :], scores, -1e9)

    weights = softmax(scores, axis=-1)

    attended = weights @ value
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch_size, sequence_length, num_hidden_dims
    )
    return attended


@profile
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


@profile
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
    mask,
    dropout_rate,
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
    attention_projection = dropout(attention_projection, dropout_rate)
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
    mlp_output = dropout(mlp_output, dropout_rate)

    second_input_residual = input_residual + mlp_output

    logits = final_logits(
        second_input_residual, token_embedding, gamma_weights3, beta_weights3
    )
    return softmax(logits, axis=-1)


if __name__ == "__main__":
    print("Starting transformer pass...")
    # Dimensions
    batch_size = 32
    sequence_length = 128
    num_hidden_dims = 768
    heads = 12
    dims_per_head = num_hidden_dims // heads
    dims_mlp = 4 * num_hidden_dims

    vocab_size = 50257  # typical GPT-style vocab

    dropout_rate = 0.1  # typical dropout rate in transformers

    # The mean (0) and std dev (0.02) match the initialisation range in
    # standard transformer implementations (GPT, BERT, T5)
    # — this range prevents exploding activations and helps training stability.

    # Each token ID and token position maps to a vector in the token embedding matrix.
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, sequence_length))
    token_embedding = np.random.normal(
        0, 0.02, size=(vocab_size, num_hidden_dims)
    ).astype(np.float32)
    positional_embedding = np.random.normal(
        0, 0.02, size=(sequence_length, num_hidden_dims)
    ).astype(np.float32)

    # Input tensor: standard normal distribution (typical in transformers)
    input_tensor = np.random.randn(batch_size, sequence_length, num_hidden_dims).astype(
        np.float32
    )

    # Learned scale (gamma) and shift (beta) for each feature
    gamma_weights1 = np.random.normal(
        loc=1.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)
    beta_weights1 = np.random.normal(
        loc=0.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)
    gamma_weights2 = np.random.normal(
        loc=1.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)
    beta_weights2 = np.random.normal(
        loc=0.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)
    gamma_weights3 = np.random.normal(
        loc=1.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)
    beta_weights3 = np.random.normal(
        loc=0.0, scale=0.02, size=(num_hidden_dims,)
    ).astype(np.float32)

    # Projection weights and bias
    projection_weights = np.random.randn(num_hidden_dims, num_hidden_dims).astype(
        np.float32
    )
    projection_bias = np.random.randn(num_hidden_dims).astype(np.float32)

    # Attention weights
    query_weights = np.random.randn(num_hidden_dims, num_hidden_dims).astype(np.float32)
    key_weights = np.random.randn(num_hidden_dims, num_hidden_dims).astype(np.float32)
    value_weights = np.random.randn(num_hidden_dims, num_hidden_dims).astype(np.float32)

    # MLP weights
    mlp_weights1 = np.random.randn(dims_mlp, num_hidden_dims).astype(np.float32)
    mlp_bias1 = np.random.randn(dims_mlp).astype(np.float32)
    mlp_weights2 = np.random.randn(num_hidden_dims, dims_mlp).astype(np.float32)
    mlp_bias2 = np.random.randn(num_hidden_dims).astype(np.float32)

    mask = causal_mask(sequence_length)[None, :, :].repeat(batch_size, axis=0)

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
        mask,
        dropout_rate
    )

    print("Tranformer pass completed")
