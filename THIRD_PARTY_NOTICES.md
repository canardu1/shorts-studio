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
