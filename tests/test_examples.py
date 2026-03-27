import requests
import pytest
import subprocess
from conftest import REPO_ROOT


def test_hello(dev_server):
    port = dev_server
    response = requests.get(f"http://localhost:{port}")
    assert response.status_code == 200
    assert (
        response.text
        == '{"message":"Hello world TEST! Version: 0.1.1 baz"}'
    )        
    assert response.headers["content-type"] == "application/json"