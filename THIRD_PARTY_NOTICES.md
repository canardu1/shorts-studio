# Third-party notices

Shorts Studio / Face Reframe is licensed under GPL-3.0 (see `LICENSE`). It uses the following third-party components, which keep their own licences.

## Models shipped in this repository (`src/face_reframe/models/`)

| Model | Use | Source | Licence |
|---|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` | face detection | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| SFace `face_recognition_sface_2021dec_int8.onnx` | face recognition (who is who) | [opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 |
| Real-ESRGAN `realesr-general-x4v3` | AI enhance | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |

## Models downloaded on first use (not redistributed here)

| Model | Use | Source | Licence |
|---|---|---|---|
| U²-Net-p `u2netp.onnx` | subject cut-out for thumbnails | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net), ONNX via [rembg](https://github.com/danielgatis/rembg) | Apache-2.0 |
| GFPGAN 1.4 | face restoration | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN), ONNX via [facefusion-assets](https://github.com/facefusion/facefusion-assets) | Apache-2.0 |
| GPEN-BFR 256 / 512 | face restoration | [yangxy/GPEN](https://github.com/yangxy/GPEN), ONNX via facefusion-assets | **non-commercial use only** |
| CodeFormer | face restoration | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer), ONNX via facefusion-assets | **S-Lab License 1.0 — non-commercial use only** |
| Whisper (faster-whisper / CTranslate2 conversion) | transcription, translation | [openai/whisper](https://github.com/openai/whisper), [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| Silero VAD `silero_vad_v6.onnx` | speech detection for silence cuts | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), bundled in faster-whisper | MIT |
| RVM `rvm_mobilenetv3_fp32.onnx` | person matte: background blur, thumbnail cut-outs | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | MIT |
| BiSeNet `bisenet_resnet_34.onnx` | face parsing: precise skin mask for beauty | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), ONNX via facefusion-assets | see source repo |
| RNNoise `sh.rnnn` | neural audio denoise (FFmpeg `arnndn`) | [richardpl/arnndn-models](https://github.com/richardpl/arnndn-models); RNNoise is BSD-3-Clause | see source repo |
| SG-LLIE `semantic_guided_llie_180x324.onnx` | low-light lift (fast) | [royshil/obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) | MIT |
| URetinex `uretinex_net_180x320.onnx` | low-light lift (quality) | royshil/obs-backgroundremoval | MIT |
| YAMNet `yamnet.onnx` + class map | audio events (laughter, applause, music) | [zeropointnine/yamnet-onnx](https://huggingface.co/zeropointnine/yamnet-onnx); map via [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 |
| ECAPA-TDNN `ecapa-any-length.onnx` | voice embeddings for speaker follow | [kenrunio/transcriber-speaker-onnx](https://huggingface.co/kenrunio/transcriber-speaker-onnx) | Apache-2.0 |
| RIFE 4.8 `rife48_ensemble_True_scale_1_sim.onnx` | frame interpolation (slow motion) | [yuvraj108c/rife-onnx](https://huggingface.co/yuvraj108c/rife-onnx); original [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | MIT |
| NanoTrackV2 `nanotrack_backbone/head_sim.onnx` | picked-object tracking (OpenCV TrackerNano) | [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers) | Apache-2.0 |
| DeepFilterNet `deep-filter` binary | neural audio denoise | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT |

If you use Shorts Studio commercially, pick **GFPGAN** (or "off") for face restoration: the GPEN and CodeFormer weights may not be used for commercial purposes.

## Libraries

| Library | Licence |
|---|---|
| OpenCV | Apache-2.0 |
| NumPy | BSD-3-Clause |
| onnxruntime | MIT |
| faster-whisper, CTranslate2 | MIT |
| PySide6 / Qt for Python | LGPL-3.0 (used as unmodified, dynamically loaded libraries) |
| imageio-ffmpeg | BSD-2-Clause; the bundled FFmpeg binary is GPL (it includes x264) |
| FFmpeg | LGPL-2.1+ / GPL-2.0+ depending on the build |
| scikit-image (tests only) | BSD-3-Clause |
