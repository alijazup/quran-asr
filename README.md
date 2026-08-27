# FastConformer Quran ASR on Replicate

Frame-accurate Quranic speech recognition and subtitle segmentation using NVIDIA FastConformer Large (`Muno459/fastconformer-quran`).

## How to Deploy to Replicate

1. Create a new GitHub repository (e.g. `fastconformer-quran`).
2. Push the files in this folder (`cog.yaml`, `predict.py`, `README.md`) to your GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/alijazup/fastconformer-quran.git
   git push -u origin main
   ```
3. Open [replicate.com/alijazup/fastconformer-quran](https://replicate.com/alijazup/fastconformer-quran) and click **"Connect GitHub repository"**.
4. Select `alijazup/fastconformer-quran`. Replicate will automatically build and publish the model version in the cloud!
