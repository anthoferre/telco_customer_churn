import pytest
from fastapi.testclient import TestClient

def test_create_item(client: TestClient):
    response = client.post(
        "/api/v1/items/",
        json={"name": "Test Item", "description": "A test item", "price": 10.0, "tax": 1.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["item"]["name"] == "Test Item"
    assert "id" in data["item"]
    assert data["item"]["id"] == 1

def test_read_item(client: TestClient):
    create_response = client.post(
        "/api/v1/items/",
        json={"name": "Another Item", "price": 20.0}
    )
    item_id = create_response.json()["item"]["id"]

    read_response = client.get(f"/api/v1/items/{item_id}")
    assert read_response.status_code == 200
    read_data = read_response.json()
    assert read_data["item"]["name"] == "Another Item"
    assert read_data["item"]["id"] == item_id

    not_found_response = client.get("/api/v1/items/999")
    assert not_found_response.status_code == 404
    assert "Item non trouvé" in not_found_response.json()["detail"]

def test_read_all_items(client: TestClient):
    client.post("/api/v1/items/", json={"name": "Item A", "price": 1.0})
    client.post("/api/v1/items/", json={"name": "Item B", "price": 2.0})
    client.post("/api/v1/items/", json={"name": "Item C", "price": 3.0})

    response = client.get("/api/v1/items/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["name"] == "Item A"

    paginated_response = client.get("/api/v1/items/?skip=1&limit=1")
    assert paginated_response.status_code == 200
    paginated_data = paginated_response.json()
    assert len(paginated_data) == 1
    assert paginated_data[0]["name"] == "Item B"

def test_update_item(client: TestClient):
    create_response = client.post("/api/v1/items/", json={"name": "Old Name", "price": 10.0})
    item_id = create_response.json()["item"]["id"]

    update_response = client.put(
        f"/api/v1/items/{item_id}",
        json={"name": "Updated Name", "description": "New description", "price": 15.0, "tax": 1.5}
    )
    assert update_response.status_code == 200
    updated_data = update_response.json()
    assert updated_data["updated_item"]["name"] == "Updated Name"
    assert updated_data["updated_item"]["description"] == "New description"
    assert updated_data["updated_item"]["price"] == 15.0

    not_found_response = client.put(
        "/api/v1/items/999",
        json={"name": "Non Existent", "price": 1.0}
    )
    assert not_found_response.status_code == 404

def test_delete_item(client: TestClient):
    create_response = client.post("/api/v1/items/", json={"name": "To Delete", "price": 5.0})
    item_id = create_response.json()["item"]["id"]

    delete_response = client.delete(f"/api/v1/items/{item_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/api/v1/items/{item_id}")
    assert get_response.status_code == 404

    not_found_response = client.delete("/api/v1/items/999")
    assert not_found_response.status_code == 404