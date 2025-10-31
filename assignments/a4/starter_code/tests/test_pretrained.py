import pytest
import torch
from load_data import load_csv_data, ModelContextualSimilarityDataset, ModelIsolatedSimilarityDataset



eval_cont_data_folder = "data/contextual_similarity"
eval_isol_data_folder = "data/isolated_similarity"


#test examples:
# Contextual: 3,car,carriage,"From 1949 The Overland moved into the modern era, with new air conditioned Corten steel <strong>carriages</strong> gradually entering service on the train. Finished in maroon, with a fluted stainless steel panel on each side of the <strong>cars</strong>, and black roof. This livery remained until the 1990s.",cars,carriages

# Isolated: 0,announcement,warning

@pytest.mark.test_gpt2_contextual
def test_gpt2_contextual():
    model_type="gpt2"
    cont_dev_x = load_csv_data(eval_cont_data_folder, "contextual_dev_x.csv")
    cont_dev_y = load_csv_data(eval_cont_data_folder, "contextual_dev_y.csv")

    # Build the dataset
    contextual_dataset=ModelContextualSimilarityDataset(model_type, cont_dev_x, cont_dev_y)
    example_instance=contextual_dataset[3]
    tgt_word1_span=torch.tensor([44,45])
    tgt_word2_span=torch.tensor([18, 20])
    assert torch.equal(example_instance[2], tgt_word1_span)
    assert torch.equal(example_instance[3], tgt_word2_span)

@pytest.mark.test_bert_contextual
def test_bert_contextual():
    model_type="bert"
    cont_dev_x = load_csv_data(eval_cont_data_folder, "contextual_dev_x.csv")
    cont_dev_y = load_csv_data(eval_cont_data_folder, "contextual_dev_y.csv")

    # Build the dataset
    contextual_dataset=ModelContextualSimilarityDataset(model_type, cont_dev_x, cont_dev_y)
    example_instance=contextual_dataset[3]
    tgt_word1_span=torch.tensor([43,44])
    tgt_word2_span=torch.tensor([19,20])
    
    assert torch.equal(example_instance[2], tgt_word1_span)
    assert torch.equal(example_instance[3], tgt_word2_span)

@pytest.mark.test_gpt2_isolated
def test_gpt2_isolated():
    model_type="gpt2"
    isol_dev_x = load_csv_data(eval_isol_data_folder, "isolated_dev_x.csv")
    isol_dev_y = load_csv_data(eval_isol_data_folder, "isolated_dev_y.csv")

    isolated_dataset = ModelIsolatedSimilarityDataset(model_type, isol_dev_x, isol_dev_y)
    example_instance=isolated_dataset[0]

    tgt_word1_span=torch.tensor([0, 3])
    tgt_word2_span=torch.tensor([0, 1])
    assert torch.equal(example_instance[2], tgt_word1_span)
    assert torch.equal(example_instance[5], tgt_word2_span)
    
@pytest.mark.test_bert_isolated
def test_bert_isolated():
    model_type="bert"
    isol_dev_x = load_csv_data(eval_isol_data_folder, "isolated_dev_x.csv")
    isol_dev_y = load_csv_data(eval_isol_data_folder, "isolated_dev_y.csv")

    isolated_dataset = ModelIsolatedSimilarityDataset(model_type, isol_dev_x, isol_dev_y)
    example_instance=isolated_dataset[0]

    tgt_word1_span=torch.tensor([1, 2])
    tgt_word2_span=torch.tensor([1, 2])
    assert torch.equal(example_instance[2], tgt_word1_span)
    assert torch.equal(example_instance[5], tgt_word2_span)