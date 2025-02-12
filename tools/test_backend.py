import httpx

# Define the base URL for the running backend
BASE_URL = "http://localhost:5000"

# Initialize the HTTP client
client = httpx.Client()

# Test 1: GET /api/video_list
def test_get_video_list():
    response = client.get(f"{BASE_URL}/api/video_list")
    print("Test 1 - GET /api/video_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())  # Print the response data

# Test 2: GET /api/video_fetch
def test_get_video_fetch():
    response = client.get(f"{BASE_URL}/api/video_fetch?video_id=1")
    print("Test 2 - GET /api/video_fetch")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 3: POST /api/product_like
def test_post_product_like():
    response = client.post(f"{BASE_URL}/api/product_like?user_id=1&product_id=1")
    print("Test 3 - POST /api/product_like")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 4: GET /api/product_like_list
def test_get_product_like_list():
    response = client.get(f"{BASE_URL}/api/product_like_list?user_id=1")
    print("Test 4 - GET /api/product_like_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 5: POST /api/product_unlike
def test_post_product_unlike():
    response = client.post(f"{BASE_URL}/api/product_unlike?user_id=1&product_id=1")
    print("Test 5 - POST /api/product_unlike")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 6: GET /api/similar_product_list
def test_get_similar_product_list():
    response = client.get(f"{BASE_URL}/api/similar_product_list")
    print("Test 6 - GET /api/similar_product_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 7: GET /api/highlight_product_list
def test_get_highlight_product_list():
    response = client.get(f"{BASE_URL}/api/highlight_product_list?video_id=1")
    print("Test 7 - GET /api/highlight_product_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Run the tests
test_get_video_list()
test_get_video_fetch()
test_post_product_like()
test_get_product_like_list()
test_post_product_unlike()
test_get_similar_product_list()
test_get_highlight_product_list()

# Close the client after all tests
client.close()
