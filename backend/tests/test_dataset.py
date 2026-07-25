def test_dataset_summary_endpoint(client):
    resp = client.get("/api/v1/dataset/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_count"] == 818238
    assert len(data["features"]) == 30
    assert "CERN/ATLAS" in data["source"]
    assert data["doi"] == "10.7483/OPENDATA.ATLAS.ZBP2.M5T8"


def test_dataset_feature_names(client):
    resp = client.get("/api/v1/dataset/summary")
    features = resp.json()["features"]
    assert "DER_mass_MMC" in features
    assert "PRI_tau_pt" in features
    assert len([f for f in features if f.startswith("PRI_")]) == 17
    assert len([f for f in features if f.startswith("DER_")]) == 13
