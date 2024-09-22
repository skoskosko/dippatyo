import yaml


with open("clean_data.yaml") as stream:
    data = yaml.safe_load(stream)

items = []

for city, images in data.items():
    for image, t in images.items():
        items.append({"city": city, "name": image, "truth_type": t["name"]})

print(f"total: {len(items)}")

print(f"failure: {len([item for item in items if item['truth_type'] in ['failure'] ])}")

print(f"focal: {len([item for item in items if item['truth_type'] in ['focal'] ])}")

print(f"vertical: {len([item for item in items if item['truth_type'] in ['vertical'] ])}")

print(f"other: {len([item for item in items if item['truth_type'] not in ['focal', 'failure', 'vertical']])}")