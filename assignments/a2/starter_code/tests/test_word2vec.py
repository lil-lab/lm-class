import pytest
import torch
from word2vec import Word2vec

@pytest.mark.test_forward_format
def test_forward():
    model = Word2vec(10, 4)
    input_tokens = torch.randint(0, 10, (20,))
    context_tokens = torch.randint(0, 10, (20,))
    negative_context = torch.randint(0, 10, (20, 5))

    input_embeds, context_embeds, negative_embeds = model(input_tokens, context_tokens, negative_context)
    
    assert input_embeds.shape == (20, 4)
    assert context_embeds.shape == (20, 4)
    assert negative_embeds.shape == (20, 5, 4)


@pytest.mark.test_pred_format
def test_pred():
    model = Word2vec(10, 4)
    input_tokens = torch.randint(0, 10, (20,))
    embeds = model.pred(input_tokens)
    assert embeds.shape == (20, 4)


@pytest.mark.test_compute_loss_no_neg_sampling
def test_compute_loss_no_neg_sampling():
    model = Word2vec(10, 4)
    input_embeds = torch.load('tests/unit_test_data/unit_test_input_embeds.pth')
    context_embeds = torch.load('tests/unit_test_data/unit_test_context_embeds.pth')

    loss = model.compute_loss(input_embeds, context_embeds)
 
    tgt_loss = torch.load('tests/unit_test_data/unit_test_tgt_no_neg_sampling_loss.pth')
    assert torch.allclose(loss, tgt_loss)

@pytest.mark.test_compute_loss_with_neg_sampling
def test_compute_loss_neg_sampling():
    torch.manual_seed(0)
    model = Word2vec(10, 4)
    input_embeds = torch.load('tests/unit_test_data/unit_test_input_embeds.pth')
    context_embeds = torch.load('tests/unit_test_data/unit_test_context_embeds.pth')
    negative_embeds = torch.load('tests/unit_test_data/unit_test_negative_embeds.pth')

    loss = model.compute_loss(input_embeds, context_embeds, negative_embeds)
    tgt_loss = torch.load('tests/unit_test_data/unit_test_tgt_w_neg_sampling_loss.pth')
    assert torch.allclose(loss, tgt_loss)