# File name: model_client.py
import requests
import time
import ray
english_text = "Hello world!"
a=[i for i in range (100)]
b=[i for i in range (100)]
begin_time=time.time()

@ray.remote
def ask(i):
    print(i)
    return requests.post("http://127.0.0.1:8000/", json=english_text)

def calculate(n):
    for i in range (n):
        a[i]=ask.remote(i)
    for i in range (n):
        b[i]=ray.get(a[i])

calculate(100)
end_time=time.time()
duration=end_time-begin_time
print(duration)