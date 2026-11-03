# BlockVerify test models

Model files for testing the BlockVerify site. Every family has an original to register and
variants to verify against it. The weights are synthetic (seeded random values at realistic
scales), not trained models. `python3 test_models/generate.py` rebuilds every file identically.

## How to test

1. **Register** each `*_original.json` on the Register tab. Use the folder's family name as the
   Model name (for example `cardio-ecg-cnn`), so batch scan can match files by name. Note each
   Model ID.
2. **Verify** each variant on the Verify tab against its family's Model ID, then press
   **Run Deep Layer Inspection** on a mismatch. The table below gives the expected verdict.
   Do this before step 3.
3. **Versions**: on the Versions tab, add `*_v2_finetuned.json` (and `v3` for the fraud model) as
   a new version. Verifying v2 then passes. The original now fails, because Verify checks the
   current version.
4. **Compare**: put an original and any variant side by side to see the layer diff.
5. **Batch**: drop all of `batch_scan/` onto Batch verification at once.
6. **Binary formats**: `07_binary-formats/` has a `.safetensors` file and a raw `.bin`. They get
   a whole-file check only, with no layer inspector.

Every register and verify writes an Algorand Testnet transaction (fee 0.001 ALGO), and each
account can make 10 writes per minute.

Known limitation: adding a version stores the new file hash but not its layer list, so the
inspector keeps comparing against v1's layers. Run the attack files (step 2) before adding
versions (step 3).

## Files

| File | Kind | What changed | Expected |
|---|---|---|---|
| `01_cardio-ecg-cnn/cardio-ecg-cnn_original.json` | original | 5253 values in 13 layers | Register with Model name `cardio-ecg-cnn`. Verify: Integrity Verified |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_v2_finetuned.json` | legit update | fine-tuned head: dense_1.weight, dense_out.weight | Add on the Versions tab, then Verify: Integrity Verified |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_single_weight.json` | tampered | conv1d_2.weight[853] nudged by 0.000001 (smallest change the file can hold) | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_backdoor_weights.json` | tampered | 3 outlier weights (about 9.0) planted in dense_1.weight at [743, 1315, 1706] | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering, 3 outlier weights listed |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_rogue_layer.json` | tampered | extra layer 'shortcut_bypass.weight' inserted before dense_out.weight | Integrity Mismatch. Inspector: CRITICAL, Topology Poisoning |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_removed_layer.json` | tampered | layer 'groupnorm_1.weight' deleted | Integrity Mismatch. Inspector: WARNING, Layer Excision |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_reordered_layers.json` | tampered | layers 'dense_1.weight' and 'dense_out.weight' swapped (same values, new order) | Integrity Mismatch. Inspector: WARNING, Layer Reordering |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_attack_metadata_only.json` | tampered | metadata edited (fake accuracy and approval added), weights untouched | Integrity Mismatch. Inspector: SECURE, every layer matches (only the file changed) |
| `01_cardio-ecg-cnn/cardio-ecg-cnn_reformatted_same_weights.json` | harmless re-save | same JSON re-saved pretty-printed; every value identical | Integrity Mismatch. Inspector: SECURE, every layer matches (only the formatting changed) |
| `02_retina-dr-resnet/retina-dr-resnet_original.json` | original | 5877 values in 14 layers | Register with Model name `retina-dr-resnet`. Verify: Integrity Verified |
| `02_retina-dr-resnet/retina-dr-resnet_v2_finetuned.json` | legit update | fine-tuned head: block2.proj.weight, head.fc.weight | Add on the Versions tab, then Verify: Integrity Verified |
| `02_retina-dr-resnet/retina-dr-resnet_attack_single_weight.json` | tampered | block1.conv2.weight[768] nudged by 0.000001 (smallest change the file can hold) | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering |
| `02_retina-dr-resnet/retina-dr-resnet_attack_backdoor_weights.json` | tampered | 3 outlier weights (about 9.0) planted in block2.proj.weight at [54, 129, 380] | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering, 3 outlier weights listed |
| `02_retina-dr-resnet/retina-dr-resnet_attack_rogue_layer.json` | tampered | extra layer 'shortcut_bypass.weight' inserted before head.fc.weight | Integrity Mismatch. Inspector: CRITICAL, Topology Poisoning |
| `02_retina-dr-resnet/retina-dr-resnet_attack_removed_layer.json` | tampered | layer 'stem.gn.weight' deleted | Integrity Mismatch. Inspector: WARNING, Layer Excision |
| `02_retina-dr-resnet/retina-dr-resnet_attack_reordered_layers.json` | tampered | layers 'block2.proj.weight' and 'head.fc.weight' swapped (same values, new order) | Integrity Mismatch. Inspector: WARNING, Layer Reordering |
| `02_retina-dr-resnet/retina-dr-resnet_attack_metadata_only.json` | tampered | metadata edited (fake accuracy and approval added), weights untouched | Integrity Mismatch. Inspector: SECURE, every layer matches (only the file changed) |
| `02_retina-dr-resnet/retina-dr-resnet_reformatted_same_weights.json` | harmless re-save | same JSON re-saved pretty-printed; every value identical | Integrity Mismatch. Inspector: SECURE, every layer matches (only the formatting changed) |
| `03_fraud-detector-mlp/fraud-detector-mlp_original.json` | original | 4157 values in 8 layers | Register with Model name `fraud-detector-mlp`. Verify: Integrity Verified |
| `03_fraud-detector-mlp/fraud-detector-mlp_v2_finetuned.json` | legit update | fine-tuned head: fc2.weight, fc_out.weight | Add on the Versions tab, then Verify: Integrity Verified |
| `03_fraud-detector-mlp/fraud-detector-mlp_v3_finetuned.json` | legit update | fine-tuned head: fc2.weight, fc_out.weight | Add on the Versions tab, then Verify: Integrity Verified |
| `03_fraud-detector-mlp/fraud-detector-mlp_attack_backdoor_weights.json` | tampered | 3 outlier weights (about 9.0) planted in fc2.weight at [664, 1109, 1230] | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering, 3 outlier weights listed |
| `03_fraud-detector-mlp/fraud-detector-mlp_attack_single_weight.json` | tampered | fc1.weight[640] nudged by 0.000001 (smallest change the file can hold) | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering |
| `04_sentiment-lstm/sentiment-lstm_original.json` | original | 5363 values in 7 layers | Register with Model name `sentiment-lstm`. Verify: Integrity Verified |
| `04_sentiment-lstm/sentiment-lstm_attack_rogue_layer.json` | tampered | extra layer 'shortcut_bypass.weight' inserted before classifier.weight | Integrity Mismatch. Inspector: CRITICAL, Topology Poisoning |
| `04_sentiment-lstm/sentiment-lstm_attack_removed_layer.json` | tampered | layer 'lstm.bias' deleted | Integrity Mismatch. Inspector: WARNING, Layer Excision |
| `04_sentiment-lstm/sentiment-lstm_attack_reordered_layers.json` | tampered | layers 'embedding.weight' and 'classifier.weight' swapped (same values, new order) | Integrity Mismatch. Inspector: WARNING, Layer Reordering |
| `05_traffic-sign-vit/traffic-sign-vit_original.json` | original | 11979 values in 15 layers | Register with Model name `traffic-sign-vit`. Verify: Integrity Verified |
| `05_traffic-sign-vit/traffic-sign-vit_attack_metadata_only.json` | tampered | metadata edited (fake accuracy and approval added), weights untouched | Integrity Mismatch. Inspector: SECURE, every layer matches (only the file changed) |
| `05_traffic-sign-vit/traffic-sign-vit_reformatted_same_weights.json` | harmless re-save | same JSON re-saved pretty-printed; every value identical | Integrity Mismatch. Inspector: SECURE, every layer matches (only the formatting changed) |
| `05_traffic-sign-vit/traffic-sign-vit_attack_single_weight.json` | tampered | block0.mlp.fc1.weight[682] nudged by 0.000001 (smallest change the file can hold) | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering |
| `06_crop-disease-mobilenet/crop-disease-mobilenet_original.json` | original | 6118 values in 13 layers | Register with Model name `crop-disease-mobilenet`. Verify: Integrity Verified |
| `06_crop-disease-mobilenet/crop-disease-mobilenet_attack_backdoor_weights.json` | tampered | 3 outlier weights (about 9.0) planted in pw2.weight at [39, 62, 1805] | Integrity Mismatch. Inspector: CRITICAL, Weight Tampering, 3 outlier weights listed |
| `06_crop-disease-mobilenet/crop-disease-mobilenet_v2_finetuned.json` | legit update | fine-tuned head: pw2.weight, classifier.weight | Add on the Versions tab, then Verify: Integrity Verified |
| `07_binary-formats/keyword-spotting.safetensors` | original | safetensors file, 4 tensors | Register with Model name `keyword-spotting`. Verify: Integrity Verified |
| `07_binary-formats/keyword-spotting_tampered.safetensors` | tampered | one float32 changed in its last bit | Integrity Mismatch (binary files have no layer inspector) |
| `07_binary-formats/anomaly-autoencoder.bin` | original | 512 raw float32 weights | Register with Model name `anomaly-autoencoder`. Verify: Integrity Verified |
| `07_binary-formats/anomaly-autoencoder_tampered.bin` | tampered | weight 300 changed in its last bit | Integrity Mismatch (binary files have no layer inspector) |
| `08_unregistered/weather-forecast-gru.json` | unregistered | a model nobody registered | Batch scan: Unregistered |
| `batch_scan/cardio-ecg-cnn.json` | batch | copy of the original | Verified |
| `batch_scan/retina-dr-resnet.json` | batch | backdoored copy under the registered name | Tampered (name matches, hash does not) |
| `batch_scan/sentiment-lstm.json` | batch | reordered copy under the registered name | Tampered (name matches, hash does not) |
| `batch_scan/traffic-sign-vit.json` | batch | copy of the original | Verified |
| `batch_scan/weather-forecast-gru.json` | batch | never registered | Unregistered |

SHA-256 of every file is in `MANIFEST.csv`.
