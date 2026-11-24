from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_has_banner():
    r = client.get('/api/health')
    assert r.status_code == 200
    j = r.json()
    assert j['status'] == 'ok'
    assert 'Research demo on recorded greenhouse data. Not for field use.' == j['banner']

def test_model_info_no_upload():
    # no upload endpoints exist
    r = client.post('/api/replay')
    assert r.status_code in (405, 404)

def test_plants_default_mixed_only():
    r = client.get('/api/plants?seed=42&mixed_only=true')
    assert r.status_code == 200
    j = r.json()
    assert j['banner']
    assert isinstance(j['plants'], list)

def test_replay_rejects_early_day_when_mixed_only():
    # day 2 is single-class early day
    r = client.get('/api/plants?seed=42&mixed_only=false')
    plants = r.json()['plants']
    pk = plants[0]['plant_key']
    r2 = client.get(f'/api/replay?seed=42&plant_key={pk}&day=2&mixed_only=true')
    assert r2.status_code == 400

def test_image_path_forbidden_outside_manifest():
    r = client.get('/api/image', params={'path': '/etc/passwd'})
    assert r.status_code == 403
