import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8012/search"

def query():
    r = requests.post(URL, json={
        "query": "tax deduction 80C HRA PPF ELSS",
        "top_k": 5
    })
    return r.status_code

def run_test(total_requests=200, concurrency=20):
    start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: query(), range(total_requests)))

    end = time.time()

    success = results.count(200)
    duration = end - start
    qps = total_requests / duration

    print("\n===== QPS TEST =====")
    print(f"Total Requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Successful: {success}")
    print(f"Time: {duration:.2f}s")
    print(f"QPS: {qps:.2f}")

if __name__ == "__main__":
    run_test()