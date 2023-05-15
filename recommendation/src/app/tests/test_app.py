from fastapi.testclient import TestClient
from app.api import my_app

client = TestClient(my_app)

def test_ping():
    """Test Ping get endpoint, here we should expect a 200 status code
    and a response json object in the form {"status": "ok"}, that means
    all the model where loaded and there is no problem"""
    response = client.get('/ping')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
