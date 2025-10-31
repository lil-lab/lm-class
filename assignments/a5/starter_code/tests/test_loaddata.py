from load_data import T5Dataset, normal_collate_fn
import torch
import pytest

@pytest.mark.test_decoder_bos
def test_collate_fn_decoder_bos():
    data_folder = 'data'
    dset = T5Dataset(data_folder, "dev")
    decoder_bos=dset[0][2][0]
    assert decoder_bos!=3  #should not be regular text tokens, such as "▁" or "SEL"
    assert decoder_bos!=23143
    for i in range(1, len(dset)):
        this_instance= dset[i]
        assert this_instance[2][0] == decoder_bos, f"Decoder BOS token mismatch at index {i}"

@pytest.mark.test_collate_encoder_inputs
def test_collate_fn_encoder_inputs_shape():
    data_point1=torch.load("tests/unit_test_data/datapoint1.pth")
    data_point1=(data_point1["encoder_ids"], data_point1["encoder_mask"],data_point1["decoder_ids"], data_point1["sql_line"])
    data_point2=torch.load("tests/unit_test_data/datapoint2.pth")
    data_point2=(data_point2["encoder_ids"], data_point2["encoder_mask"],data_point2["decoder_ids"], data_point2["sql_line"])
    batch=[data_point1, data_point2]
    collated=normal_collate_fn(batch)

    assert collated[0].shape[0] == 2  # batch size
    assert collated[0].shape[1] == 15 #pad to max length of the batch

@pytest.mark.test_collate_decoder_inputs
def test_collate_fn_decoder_inputs():
    data_point1=torch.load("tests/unit_test_data/datapoint1.pth")
    data_point1=(data_point1["encoder_ids"], data_point1["encoder_mask"],data_point1["decoder_ids"], data_point1["sql_line"])
    data_point2=torch.load("tests/unit_test_data/datapoint2.pth")
    data_point2=(data_point2["encoder_ids"], data_point2["encoder_mask"],data_point2["decoder_ids"], data_point2["sql_line"])
    batch=[data_point1, data_point2]
    collated=normal_collate_fn(batch)

    decoder_input=torch.load("tests/unit_test_data/decoder_inputs.pth")
    assert torch.equal(collated[2], decoder_input)

@pytest.mark.test_collate_decoder_targets
def test_collate_fn_decoder_targets():
    data_point1=torch.load("tests/unit_test_data/datapoint1.pth")
    data_point1=(data_point1["encoder_ids"], data_point1["encoder_mask"],data_point1["decoder_ids"], data_point1["sql_line"])
    data_point2=torch.load("tests/unit_test_data/datapoint2.pth")
    data_point2=(data_point2["encoder_ids"], data_point2["encoder_mask"],data_point2["decoder_ids"], data_point2["sql_line"])
    batch=[data_point1, data_point2]
    collated=normal_collate_fn(batch)

    decoder_target=torch.load("tests/unit_test_data/decoder_targets.pth")
    assert torch.equal(collated[3], decoder_target)

@pytest.mark.test_collate_decoder_initial_inputs
def test_collate_fn_decoder_initial_decoder_inputs():
    data_point1=torch.load("tests/unit_test_data/datapoint1.pth")
    data_point1=(data_point1["encoder_ids"], data_point1["encoder_mask"],data_point1["decoder_ids"], data_point1["sql_line"])
    data_point2=torch.load("tests/unit_test_data/datapoint2.pth")
    data_point2=(data_point2["encoder_ids"], data_point2["encoder_mask"],data_point2["decoder_ids"], data_point2["sql_line"])
    batch=[data_point1, data_point2]
    collated=normal_collate_fn(batch)
    initial_decoder_inputs=torch.load("tests/unit_test_data/decoder_initial_inputs.pth")
    assert torch.equal(collated[4], initial_decoder_inputs)

