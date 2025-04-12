# 🤖 3VO Dual AI Automation 🎨🔊

An advanced AI-powered web app that combines **image generation** and **voice cloning** into a single intuitive interface. Built with [Gradio](https://gradio.app/), it enables users to create personalized visuals and clone voices using text input and audio references.

---

## 🚀 Features

### 🎨 Image Generation (DALL·E 3 XL style)

- Generate stunning images from detailed prompts.
- Built-in **Prompt Builder** with customizable options:
  - Gender + age, skin tone, hairstyle, style, accessories, expressions, and celebrity-inspired flair.
- Adjustable settings:
  - Resolution (width/height), guidance scale, random seed, and negative prompts.
- Example prompts provided for quick testing.

### 🔊 Voice Cloning (XTTS v2)

- Clone any voice from a short `.wav` or `.mp3` reference.
- Multilingual text-to-speech output (supports `en`, `fr`, `es`, `ar`, `zh`, `ja`, etc.).
- Realistic and natural speech synthesis powered by XTTS v2.

---

## 🧱 Built With

- [Gradio](https://gradio.app/) – for building the interactive user interface
- [Diffusers](https://huggingface.co/docs/diffusers/index) – for image generation using Stable Diffusion XL
- [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS) – for multilingual voice cloning
- [PyTorch](https://pytorch.org/) – for GPU-accelerated inference
- [Pydub](https://github.com/jiaaro/pydub) – for audio processing
- [PIL](https://pillow.readthedocs.io/) – for saving images

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ✅ Make sure PyTorch and a compatible CUDA setup are installed for GPU acceleration.

### 3. Run the app

```bash
python app.py
```

You can also run this inside **Google Colab** or **Jupyter Notebook** if preferred.

---

## 📁 Project Structure

```
├── app.py                   # Main script with UI logic
├── /content/audio/         # Example voice files
├── output.wav              # Generated audio output
├── *.jpg                   # Generated image output
├── README.md
```

---

## 📸 Interface Preview

Coming soon...

---

## 🙌 Created by

**Youniss Jaafil**  
Software Engineer @ [3VO](https://3vo.me/)
