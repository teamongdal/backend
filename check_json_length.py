import json

filename = input()

with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    print("Number of entries:", len(data))
else:
    print("The JSON file does not contain a list. It contains:", type(data))


if isinstance(data, list):
    print("Number of entries:", len(data))
    total_clothes = 0
    for entry in data:
        clothes = entry.get("clothes")
        if isinstance(clothes, list):
            total_clothes += len(clothes)
    print("Total number of 'clothes' items:", total_clothes)
else:
    print("The JSON file does not contain a list. It contains:", type(data))