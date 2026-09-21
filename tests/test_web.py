"""API lifecycle and real band-to-browser previews (optional web extras)."""

import io
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from hmsm.web.server import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def create(client):
    response = client.post(
        "/api/jobs", json={"filename": "scan.tif", "profile": "phonola"}
    )
    assert response.status_code == 201
    return response.json()["id"]


def wait_for_job(client, identifier):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{identifier}").json()
        if state["status"] in ("complete", "error"):
            return state
        time.sleep(0.02)
    pytest.fail("Digitization did not finish")


def test_local_origin_and_validation(client):
    assert "phonola" in client.get("/api/profiles").json()
    assert (
        client.post("/api/jobs", json={"filename": "scan", "tempo": 0}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/jobs", json={"filename": "scan", "profile": "/tmp/private.json"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/jobs",
            headers={"Origin": "https://elsewhere.example"},
            json={"filename": "scan"},
        ).status_code
        == 403
    )
    assert (
        client.get("/api/profiles", headers={"Host": "elsewhere.example"}).status_code
        == 400
    )


def test_session_exclusion_and_failed_scan_cleanup(client):
    identifier = create(client)
    assert client.post("/api/jobs", json={"filename": "other"}).status_code == 409
    assert client.get(f"/api/jobs/{identifier}/midi").status_code == 409
    assert (
        client.put(f"/api/jobs/{identifier}/scan", content=b"not an image").status_code
        == 200
    )
    state = wait_for_job(client, identifier)
    assert state["status"] == "error"
    assert not (client.app.state.job.directory / "scan").exists()
    assert (
        client.put(f"/api/jobs/{identifier}/scan", content=b"again").status_code == 409
    )
    assert client.delete(f"/api/jobs/{identifier}").status_code == 200
    assert client.get(f"/api/jobs/{identifier}").status_code == 404
    create(client)


def test_oversize_upload_rejected_and_removed(client, monkeypatch):
    monkeypatch.setattr("hmsm.web.server.MAX_UPLOAD", 8)
    identifier = create(client)
    assert (
        client.put(f"/api/jobs/{identifier}/scan", content=b"123456789").status_code
        == 413
    )
    assert not (client.app.state.job.directory / "scan").exists()


def test_real_scan_tiles_notes_and_midi(client, roll_scan):
    import mido
    from hmsm.rolls import RollDigitizer
    from hmsm.profiles import load_profile

    identifier = create(client)
    with roll_scan.open("rb") as scan:
        assert (
            client.put(f"/api/jobs/{identifier}/scan", content=scan).status_code == 200
        )
    state = wait_for_job(client, identifier)
    assert state["status"] == "complete", state.get("error")
    assert len(state["tiles"]) >= 1
    assert state["safe_row"] > state["origin_row"]
    assert client.get(f"/api/jobs/{identifier}?version={state['version']}").json() == {
        "version": state["version"]
    }
    tile = client.get(f"/api/jobs/{identifier}/tiles/0.jpg")
    assert tile.status_code == 200 and tile.content.startswith(b"\xff\xd8")
    assert client.get(f"/api/jobs/{identifier}/tiles/-1.jpg").status_code == 404
    rendered = client.get(f"/api/jobs/{identifier}/midi")
    midi = mido.MidiFile(file=io.BytesIO(rendered.content))
    assert any(message.type == "note_on" for message in midi.tracks[0])
    reference = RollDigitizer(load_profile("phonola", "roll")).run(str(roll_scan))
    actual = np.array(state["notes"])
    actual[:, :2] -= state["origin_row"]
    np.testing.assert_array_equal(actual, reference.notes)
    assert client.delete(f"/api/jobs/{identifier}").status_code == 200
