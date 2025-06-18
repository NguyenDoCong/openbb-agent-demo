def search(query):
    import requests
    params = {
    "query": query
    }
    r = requests.post('http://127.0.0.1:8000/search',params=params)
    return r.json()

if __name__ == "__main__":
    result = search("Giá cổ phiếu FPT")
    print("============")
    print("Final Answer")
    print("============")
    print(result)