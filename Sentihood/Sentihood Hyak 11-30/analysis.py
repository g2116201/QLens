import torch
from transformers import GPT2Tokenizer
from utils import GPT2EmbeddingUtility

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<PAD>', 'cls_token': '<CLS>'})
embeddings = GPT2EmbeddingUtility(tokenizer)

"""
get_intermediate_probs() returns intermediate probabilities extracted by Tuned Lens

Args:
    dataset (SentihoodDataset): Dataset containing instance
                                to derive probabilites from
    example_id (int): Index of dataset instance to derive probabilites from
    embedding_model (EmbeddingModel): Submodel used to extract base Transformer's
                                      compressed embeddings.
    attention_model (AttentionModel): Submodel used to extract base Transformer's
                                      residual stream after self-attention.
    lens (TunedLens): Tuned Lens used to extract probabilities
    position (str): Either 'embedding' or 'attention'; specifies which submodel's
                    output should be passed to the inputted Tuned Lens

Returns:
    indermediate_probs (torch.Tensor): intermediate probabilities extracted by
                                       the Tuned Lens
"""
def get_intermediate_probs(dataset, example_id, embedding_model, attention_model, lens, position):
  # Deriving input tokens and their GPT-2 embeddings
  input_ids = dataset[example_id]['input_ids'].to(device)
  gpt2_embeddings = embeddings.get_embeddings(input_ids).to(device)

  # Determining which submodel to use based on inputted position
  if position == 'embedding':
    embedding = embedding_model(gpt2_embeddings)
    intermediate_logits = lens(embedding)
  elif position == 'attention':
    post_attention = attention_model(gpt2_embeddings)
    intermediate_logits = lens(post_attention)

  # Normalizing logits into probabilities via softmax
  intermediate_probs = nn.functional.softmax(intermediate_logits, dim = -1)

  return intermediate_probs


"""
get_final_probs() computes output probabilities from pretrained Transformer model

Args:
    dataset (SentihoodDataset): Dataset containing instance
                                to derive probabilites from
    example_id (int): Index of dataset instance to derive probabilites from
    final_model (TransformerModel): The base Transformer model.

Returns:
    final_probs (torch.Tensor): output probabilities for instance stemming from
                                original model
"""
def get_final_probs(dataset, example_id, final_model):
  # Deriving input tokens and their GPT-2 embeddings
  input_ids = dataset[example_id]['input_ids'].to(device)
  gpt2_embeddings = embeddings.get_embeddings(input_ids).to(device)

  # Passing GPT-2 embeddings through model to obtain class probabilities
  final_probs = final_model(gpt2_embeddings)

  return final_probs

"""
construct_state_ket() creates a state vector to represent the input
classification probabilities, where each element is the probability amplitude
(square root of the probability) for a class.

Args:
    probs_tensor (torch.Tensor): A PyTorch tensor containing the class
                                 probabilities output by the model. It is
                                 expected to have shape (1, N) or (N), where N
                                 is the number of classes.

Returns:
    ket (numpy.ndarray): Quantum ket with representing input probabilities.
"""
def construct_state_ket(probs_tensor):
  probs_vector = probs_tensor.squeeze().detach().cpu().numpy()
  ket = np.sqrt(probs_vector)

  return ket

"""
get_reflecting_plane_normal_vec() calculates normalized Householder vector
that reflects ket phi_0 onto ket phi_1.

Args:
    phi_0 (numpy.ndarray): The initial ket
    phi_1 (numpy.ndarray): The target ket that phi_1 is to be reflected onto.

Returns:
    numpy.ndarray: The normalized Householder vector (normal vector to the
                   reflecting plane)
"""
def get_reflecting_plane_normal_vec(phi_0, phi_1):
  return ((phi_1 - phi_0) / np.linalg.norm((phi_1 - phi_0)))

# get_unitary() creates a unitary Householder tranformation that maps the first
# state vector to the second

"""
get_unitary() creates a unitary Householder transformation that maps the first
state vector to the second up to a global phase.

Args:
    phi_0 (numpy.ndarray): The initial state vector
    phi_1 (numpy.ndarray): The target state vector

Returns:
    unitary (numpy.ndarray): The Householder unitary matrix ($U = I - 2vv^T$)
                             which performs the reflection.
    reflecting_plane_normal_vec (numpy.ndarray): The Householder reflection
                                                 vector used to define the
                                                 reflection.
"""
def get_unitary(phi_0, phi_1):
  identity = np.identity(phi_0.size)

  reflecting_plane_normal_vec = get_reflecting_plane_normal_vec(phi_0, phi_1)
  outer_product = np.outer(reflecting_plane_normal_vec, reflecting_plane_normal_vec)

  unitary = identity - (2 * outer_product)

  return unitary, reflecting_plane_normal_vec


"""
get_unitary_from_ex() creates a unitary Householder transformation and its
relfection vector that maps an initial state vector (phi_0) to a
target state vector (phi_1) for a specific dataset example.

Args:
    dataset (SentihoodDataset): Dataset containing the instance to analyze.
    example_id (int): Index of the dataset instance to derive state vectors from.
    embedding_model (EmbeddingModel): Submodel used to extract base Transformer's
                                      compressed embeddings.
    attention_model (AttentionModel): Submodel used to extract base Transformer's
                                      residual stream after self-attention.
    embedding_lens (TunedLens): Tuned Lens trained on residuals from base Transformer's
                                embedding layer.
    attention_lens (TunedLens): Tuned Lens trained on residuals after the base Transformer's
                                attention layer.
    final_model (TransformerModel): The base Transformer model.
    pos_1 (str): Either 'embedding' or 'attention'; specifies the output of layer
                 from which the initial state vector is derived.
    pos_2 (str, optional): Either 'attention' or 'mlp' (default); specifies the
                           layer/position for the target state.
    verbose (bool, optional): If True, prints the initial and target state vectors

Returns:
    unitary (numpy.ndarray): The Householder unitary matrix that maps the initial
                             instance-derived state vector to its target.
    hh_vec (numpy.ndarray): The normalized Householder vector of the transformation.
"""
def get_unitary_from_ex(dataset, example_id, embedding_model, attention_model, embedding_lens, attention_lens, final_model, pos_1, pos_2 = 'mlp', verbose = False):

  # Obtaining intermediate probabilities
  if pos_1 == 'embedding':
    phi_0_probs = get_intermediate_probs(dataset, example_id, embedding_model, attention_model, embedding_lens, pos_1)
    if pos_2 == 'attention':
      phi_1_probs = get_intermediate_probs(dataset, example_id, embedding_model, attention_model, attention_lens, pos_2)
    else:
      phi_1_probs = get_final_probs(dataset, example_id, final_model)
  elif pos_1 == 'attention':
    phi_0_probs = get_intermediate_probs(dataset, example_id, embedding_model, attention_model, attention_lens, pos_1)
    phi_1_probs = get_final_probs(dataset, example_id, final_model)

  # Constructing state vectors
  phi_0 = construct_state_ket(phi_0_probs)
  phi_1 = construct_state_ket(phi_1_probs)

  if verbose:
    print(f"Phi_0: \n{phi_0}")
    print(f"Phi_1: \n{phi_1}")
    print("")

  # Obtaining unitary and Householder vector
  unitary, hh_vec = get_unitary(phi_0, phi_1)

  # Checking to ensure unitary appropriately maps initial state vector to target
  if not np.allclose((unitary @ phi_0), phi_1, atol = 0.01):
    print(f"\nUnitary Constructed Incorrectly\nUnitary @ Phi_0: {(unitary @ phi_0)}\nPhi_1: {phi_1}")

  return unitary, hh_vec