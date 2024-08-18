import pytest

from transformer import CharacterLevelTransformer
import torch

@pytest.mark.transformer_1
def test_initialization():
    # Can we load the state dictionary?
    model = CharacterLevelTransformer(2, 64, 4, 128, 0.1, 36)
    state_dict = torch.load('tests/unit_test_transformer_data/unit_test_state_dict.pth')
    model.load_state_dict(state_dict)

@pytest.mark.transformer_2
def test_forward():
    # Can the model go through the forward pass without an error?
    model = CharacterLevelTransformer(2, 64, 4, 128, 0.1, 36)
    state_dict = torch.load('tests/unit_test_transformer_data/unit_test_state_dict.pth')
    model.load_state_dict(state_dict)
    model.eval()

    input_tokens, _ = torch.load('tests/unit_test_transformer_data/testing_data.pth')
    model.forward(input_tokens)

@pytest.mark.transformer_3
def test_log_probabilities():
    # Load model
    model = CharacterLevelTransformer(2, 64, 4, 128, 0.1, 36)
    state_dict = torch.load('tests/unit_test_transformer_data/unit_test_state_dict.pth')
    model.load_state_dict(state_dict)
    model.eval()

    # Compute log-probabilities in base 2 (to be used in perplexity)
    input_tokens, target_tokens = torch.load('tests/unit_test_transformer_data/testing_data.pth')
    log_probabilities = model.log_probability(input_tokens, target_tokens, base=2)
    target_log_probabilities = torch.load('tests/unit_test_transformer_data/target_log_probs.pth')

    diff = torch.mean(torch.abs(target_log_probabilities - log_probabilities)).item()
    assert(diff < 2e-4)
