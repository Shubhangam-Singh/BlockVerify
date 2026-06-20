import json, random

def generate():
    print("Generating Academic Demo Models for BlockVerify...")
    random.seed(42)  # Make models deterministic so multiple runs don't break verification
    
    # 1. Generate base architecture
    base_model = {
        "metadata": {
            "name": "ResNet-50-Medical",
            "framework": "TensorFlow.js",
            "precision": "float32",
            "params": 25600000
        },
        "layers": {
            "conv2d_1": [round(random.uniform(-1, 1), 6) for _ in range(500)],
            "batch_normalization_1": [round(random.uniform(0, 1), 6) for _ in range(100)],
            "activation_relu_1": [],
            "max_pooling2d_1": [],
            "conv2d_2": [round(random.uniform(-1, 1), 6) for _ in range(500)],
            "dense_1": [round(random.uniform(-1, 1), 6) for _ in range(256)],
            "dense_output": [round(random.uniform(-1, 1), 6) for _ in range(10)]
        }
    }

    # Write original model
    with open("demo_original.json", "w") as f:
        json.dump(base_model, f)
    print("✅ Created demo_original.json")

    # 2. Tamper the model (simulate a backdoor in dense_1)
    tampered_model = json.loads(json.dumps(base_model))
    tampered_model["layers"]["dense_1"][42] = 999.999999
    
    # Write tampered model
    with open("demo_tampered_weights.json", "w") as f:
        json.dump(tampered_model, f)
    print("🚨 Created demo_tampered_weights.json (Backdoor injected in 'dense_1')")

    # 3. Structural Anomaly (Rogue Layer)
    rogue_model = json.loads(json.dumps(base_model))
    
    # We rebuild the layers dict to insert a rogue layer right before 'dense_1'
    new_layers = {}
    for k, v in rogue_model["layers"].items():
        if k == "dense_1":
            new_layers["backdoor_bypass_layer"] = [round(random.uniform(-1, 1), 6) for _ in range(128)]
        new_layers[k] = v
        
    rogue_model["layers"] = new_layers
    rogue_model["metadata"]["params"] += 128

    with open("demo_rogue_layer.json", "w") as f:
        json.dump(rogue_model, f)
    print("🧨 Created demo_rogue_layer.json (Topology compromised: 'backdoor_bypass_layer' inserted)")

    # 4. Layer Excision (a signed layer is deleted from the model)
    excised_model = json.loads(json.dumps(base_model))
    removed = excised_model["layers"].pop("batch_normalization_1")
    excised_model["metadata"]["params"] -= len(removed)
    with open("demo_excised_layer.json", "w") as f:
        json.dump(excised_model, f)
    print("✂️  Created demo_excised_layer.json (Layer Excision: 'batch_normalization_1' removed)")

    # 5. Layer Reordering (same layers, altered execution-graph order)
    reordered_model = json.loads(json.dumps(base_model))
    items = list(reordered_model["layers"].items())
    # swap the two dense layers' positions to alter the data-flow path
    names = [k for k, _ in items]
    i1, i2 = names.index("dense_1"), names.index("dense_output")
    items[i1], items[i2] = items[i2], items[i1]
    reordered_model["layers"] = dict(items)
    with open("demo_reordered.json", "w") as f:
        json.dump(reordered_model, f)
    print("🔀 Created demo_reordered.json (Layer Reordering: 'dense_1' <-> 'dense_output')")

if __name__ == "__main__":
    generate()
    print("Done! Demonstrates 4 tampering classes: Weight Poisoning, Topology Poisoning, Layer Excision & Reordering.")
