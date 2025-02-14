import httpx

# Define the base URL for the running backend
BASE_URL = "http://localhost:8000"

# Initialize the HTTP client
client = httpx.Client()

# Test 1: GET /api/video_list
def test_get_video_list():
    response = client.get(f"{BASE_URL}/api/video_list?user_id=user_0001")
    print("Test 1 - GET /api/video_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 2: GET /api/video_play
def test_get_video_play():
    response = client.get(f"{BASE_URL}/api/video_play?video_id=video_0001")
    print("Test 2 - GET /api/video_play")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 3: GET /api/search_product
def test_search_product():
    try:
        # Sample audio file ("왼쪽 옷 정보 알려줘")
        # audio_file_path = r"C:\github\ongdal\backend\tools\sample2.wav"
        audio_file_path = r"./tools/sample2.wav"

        # Open the .wav file in binary mode for uploading
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "audio": ("sample.wav", audio_file, "audio/wav")
            }
            params = {"user_id": 1}  # The user_id as a query parameter

            # Send the POST request to the backend with the user_id and audio file
            response = client.post(f"{BASE_URL}/api/search_product", params=params, files=files)

        print("Test 3 - POST /api/search_product")
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

    except Exception as e:
        print(f"Error in Test 3: {e}")

# Test 4: GET /api/product_list
def test_get_product_list():
    response = client.get(f"{BASE_URL}/api/product_list?user_id=user_0001")
    print("Test 4 - GET /api/product_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 5: POST /api/product_like
def test_post_product_like():
    response = client.post(f"{BASE_URL}/api/product_like?user_id=user_0001&product_code=hoodie_0279")
    print("Test 5 - POST /api/product_like")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 6: POST /api/product_unlike
def test_post_product_unlike():
    response = client.post(f"{BASE_URL}/api/product_unlike?user_id=user_0001&product_code=hoodie_0279")
    print("Test 6 - POST /api/product_unlike")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 7: GET /api/product_like_list
def test_get_product_like_list():
    response = client.get(f"{BASE_URL}/api/product_like_list?user_id=user_0001")
    print("Test 7 - GET /api/product_like_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Test 8: GET /api/all_product_list
def test_get_all_product_list():
    response = client.get(f"{BASE_URL}/api/all_product_list?video_id=video_0001")
    print("Test 8 - GET /api/all_product_list")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

# Run the tests
test_get_video_list()
test_get_video_play()
test_search_product()
test_get_product_list()
test_post_product_like()
test_post_product_unlike()
test_get_product_like_list()
test_get_all_product_list()

# Close the client after all tests
client.close()

# another method of sending APIs
# # Test 1: GET /api/video_list
# def test_get_video_list():
#     params = {"user_id": "0001"}
#     response = client.get(f"{BASE_URL}/api/video_list", params=params)
#     print("Test 1 - GET /api/video_list")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 2: GET /api/video_play
# def test_get_video_play():
#     params = {"video_id": "0001"}
#     response = client.get(f"{BASE_URL}/api/video_play", params=params)
#     print("Test 2 - GET /api/video_play")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 3: GET /api/search_product
# def test_search_product():
#     try:
#         # Sample audio file ("왼쪽 옷 정보 알려줘")
#         audio_file_path = r"./tools/sample2.wav"
        
#         # Open the .wav file in binary mode for uploading
#         with open(audio_file_path, "rb") as audio_file:
#             files = {
#                 "audio": ("sample.wav", audio_file, "audio/wav")
#             }
#             params = {"user_id": "1"}  # Pass user_id as a string

#             # Send the POST request to the backend with the user_id and audio file
#             response = client.post(f"{BASE_URL}/api/search_product", params=params, files=files)

#         print("Test 3 - POST /api/search_product")
#         print("Status Code:", response.status_code)
#         print("Response JSON:", response.json())

#     except Exception as e:
#         print(f"Error in Test 3: {e}")

# # Test 4: GET /api/product_list
# def test_get_product_list():
#     params = {"user_id": "0001"}
#     response = client.get(f"{BASE_URL}/api/product_list", params=params)
#     print("Test 4 - GET /api/product_list")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 5: POST /api/product_like
# def test_post_product_like():
#     params = {"user_id": "0001", "product_code": "hoodie_0279"}
#     response = client.post(f"{BASE_URL}/api/product_like", params=params)
#     print("Test 5 - POST /api/product_like")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 6: POST /api/product_unlike
# def test_post_product_unlike():
#     params = {"user_id": "0001", "product_id": "0001"}
#     response = client.post(f"{BASE_URL}/api/product_unlike", params=params)
#     print("Test 6 - POST /api/product_unlike")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 7: GET /api/product_like_list
# def test_get_product_like_list():
#     params = {"user_id": "0001"}
#     response = client.get(f"{BASE_URL}/api/product_like_list", params=params)
#     print("Test 7 - GET /api/product_like_list")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

# # Test 8: GET /api/all_product_list
# def test_get_all_product_list():
#     params = {"video_id": "0001"}
#     response = client.get(f"{BASE_URL}/api/all_product_list", params=params)
#     print("Test 8 - GET /api/all_product_list")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())
