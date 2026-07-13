import pytest
from petri.yaml import load_yaml

def test_load_yaml():
    # Test loading petri.yaml with 13-agent roster
    load_yaml('path/to/petri.yaml')