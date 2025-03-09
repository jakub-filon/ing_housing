import os

folders = [
    "data/macro",
    "data/real_estate",
    "data/train_test"
    "models",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folder structure created successfully!")