import tempfile

from utils import DataPoint, read_labeled_data, read_unlabeled_data


def test_read_labeled_data():
    data = """id,text
1,hello
2,world
"""
    labels = """id,label
1,1
2,0
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as data_file:
        data_file.write(data)
    with tempfile.NamedTemporaryFile("w", delete=False) as labels_file:
        labels_file.write(labels)
    result = read_labeled_data(data_file.name, labels_file.name)
    assert result == [
        DataPoint(id=1, text="hello", label="1"),
        DataPoint(id=2, text="world", label="0"),
    ]


def test_read_unlabeld_data():
    data = """id,text
1,hello
2,world
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as data_file:
        data_file.write(data)
    result = read_unlabeled_data(data_file.name)
    assert result == [
        DataPoint(id=1, text="hello", label=None),
        DataPoint(id=2, text="world", label=None),
    ]
