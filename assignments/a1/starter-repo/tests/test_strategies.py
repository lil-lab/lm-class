from hypothesis import settings
from hypothesis.strategies import integers, floats


settings.register_profile("ci", max_examples=10, deadline=None)
settings.load_profile("ci")

m_ints_classes = integers(min_value=2, max_value=20)
m_ints_batch_size = integers(min_value=1, max_value=64)
l_ints = integers(min_value=2, max_value=150000)
l_int_sequence_length = integers(min_value=1, max_value=1000)