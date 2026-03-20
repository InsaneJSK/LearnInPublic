import requests

database = {
    1: "Alice",
    2: "Bob",
    3: "Charlie"
}

def get_user(user_id: int) -> str:
    return database.get(user_id, "Unknown User")

def fetch_data_from_api(url: str = "http://jsonplaceholder.typicode.com/users") -> dict:
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise requests.HTTPError(f"Failed to fetch data: {response.status_code}")