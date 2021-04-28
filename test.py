import main
from flask import Flask, request, jsonify

with main.app.test_client() as c:
    # Check if test is working
    resp = c.get('/')
    assert resp.data.decode("utf-8") == 'Hello, World!'

    resp = c.get('/load')
    assert resp.data.decode('utf-8') == 'Finish!'

    resp = c.get('/data')
    assert resp.get_json() is not None

    resp = c.get('/train')
    assert resp.data.decode('utf-8') == 'Training Complete!'

    resp = c.get('/predict')
    assert resp.get_json() is not None