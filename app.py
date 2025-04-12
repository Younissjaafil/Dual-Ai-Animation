import gradio as gr
import os
import torch
import uuid
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from TTS.api import TTS
from torch.serialization import safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

os.environ["COQUI_TOS_AGREED"] = "1"

# -----------------------------
# IMAGE GENERATION COMPONENT
# -----------------------------
DESCRIPTION = "# 3VO  Ai Agent Creation"
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "fluently/Fluently-XL-Final",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("ehristoforu/dalle-3-xl-v2", weight_name="dalle-3-xl-lora-v2.safetensors", adapter_name="dalle")
    pipe.set_adapters("dalle")
    pipe.to("cuda")

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize_seed else seed

def generate_image(prompt, negative_prompt, use_negative_prompt, seed, width, height, guidance_scale, randomize_seed):
    seed = randomize_seed_fn(seed, randomize_seed)
    if not use_negative_prompt:
        negative_prompt = ""

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil",
    ).images

    path = f"{uuid.uuid4().hex}.jpg"
    images[0].save(path, format="JPEG")
    return [path], seed

# -----------------------------
# VOICE CLONE COMPONENT
# -----------------------------
with safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clone_voice(text, audio, language):
    if audio is None or not os.path.exists(audio):
        return "Please upload a valid voice sample."

    if audio.endswith(".mp3"):
        converted = audio.replace(".mp3", ".wav")
        AudioSegment.from_mp3(audio).export(converted, format="wav")
        audio = converted

    out_path = "./output.wav"
    tts.tts_to_file(text=text, speaker_wav=audio, language=language, file_path=out_path)
    return out_path

# -----------------------------
# UI LAYOUT
# -----------------------------
with gr.Blocks(css=".gradio-container{max-width: 720px !important}") as demo:
    gr.Markdown(DESCRIPTION)

    # -- Image Generator UI --
    with gr.Tab("Image Generation"):
        with gr.Accordion("🧠 Prompt Builder (click to auto-fill prompt)", open=False):
            with gr.Row():
                gender_age = gr.Dropdown(label="👤 Gender + Age", choices=[
                    "cartoon robot girl", "old robot man", "robot baby"
                ], interactive=True)

                skin_tone = gr.Dropdown(label="🧴 Skin Tone", choices=[
                    "orange skin tone", "brown complexion", "light skin"
                ], interactive=True)

            with gr.Row():
                hair = gr.Dropdown(label="💇 Hair", choices=[
                    "long curly hair", "blonde spiky hair", "bald"
                ], interactive=True)

                style = gr.Dropdown(label="🎨 Style", choices=[
                    "Pixar style", "anime style", "cyberpunk"
                ], interactive=True)

            with gr.Row():
                accessories = gr.Dropdown(label="🎧 Accessories", choices=[
                    "wearing glasses", "with headphones", "with a red tie"
                ], interactive=True)

                expression = gr.Dropdown(label="😎 Expression", choices=[
                    "smiling", "confident", "angry", "winking"
                ], interactive=True)

            with gr.Row():
                celebrity_flavor = gr.Dropdown(label="🌟 Celebrity Flavor", choices=[
                    "looks like a famous rapper", "presidential style", "inspired by Elon"
                ], interactive=True)

            build_button = gr.Button("✨ Generate Prompt")

        prompt = gr.Text(label="Prompt")
        build_button.click(
            fn=lambda *s: ", ".join([x for x in s if x]),
            inputs=[gender_age, skin_tone, hair, style, accessories, expression, celebrity_flavor],
            outputs=prompt
        )

        with gr.Row():
            run = gr.Button("Generate")

        result = gr.Gallery(label="Result", columns=1)
        
        with gr.Accordion("Advanced Settings", open=False):
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy...",
            )
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            width = gr.Slider(label="Width", minimum=512, maximum=2048, step=8, value=1024)
            height = gr.Slider(label="Height", minimum=512, maximum=2048, step=8, value=1024)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=20.0, step=0.1, value=6)

        run.click(
            generate_image,
            inputs=[prompt, negative_prompt, use_negative_prompt, seed, width, height, guidance_scale, randomize_seed],
            outputs=[result, seed]
        )

        gr.Examples(
            examples=[
                "cartoon robot man with brown hair and headphones, waving hand, silver futuristic armor, glowing blue core, cute face, smiling, Pixar style",
                "cartoon robot with blonde hair, orange skin tone, red tie, waving hand, cute robot body, blue glowing core, confident smile, Pixar style",
                "an astronaut riding a horse in space",
                "a cartoon of a boy playing with a tiger",
                "a cute robot artist painting on an easel, concept art",
                "a woman with a holographic nemesis headpiece, concept art"
            ],
            inputs=prompt
        )

    # -- Voice Clone UI --
    with gr.Tab("Voice Cloning"):
        gr.Markdown("Upload your voice and type text to synthesize.")
        voice_text = gr.Textbox(label='Text')
        voice_input = gr.Audio(type='filepath', label='Voice Reference (.wav/.mp3)')
        language = gr.Dropdown(["en", "fr", "es", "de", "it", "pl", "ar", "zh", "ru", "ja"], value="en", label="Language")
        voice_btn = gr.Button("Clone Voice")
        voice_output = gr.Audio(label="Cloned Audio", type='filepath')

        voice_btn.click(clone_voice, inputs=[voice_text, voice_input, language], outputs=voice_output)

        gr.Examples(
            examples=[
                ["Hey! It's me Dorthy, from the Wizard of Oz. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Wizard-of-Oz-Dorthy.wav", "en"],
                ["It's me Vito Corleone, from the Godfather. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Godfather.wav", "en"],
                ["Hey, it's me Paris Hilton. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Paris-Hilton.mp3", "en"],
                ["Hey, it's me Megan Fox from Transformers. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Megan-Fox.mp3", "en"],
                ["Hey there, it's me Jeff Goldblum. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Jeff-Goldblum.mp3", "en"],
                ["Hey there, it's me Heath Ledger as the Joker. Type in whatever you'd like me to say.", "/content/Dual-Ai-Animation/audio/Heath-Ledger.mp3", "en"]
            ],
            inputs=[voice_text, voice_input, language]
        )

demo.queue(max_size=1000).launch(share=True, debug=True)
