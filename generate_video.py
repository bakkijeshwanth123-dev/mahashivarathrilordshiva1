from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_ai_client():
    """Returns an AI client preferring OpenRouter if available"""
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
        )
    return None

def get_gemini_vision(prompt):
    """Uses OpenRouter (preferred) or Gemini to generate cinematic text"""
    chat_prompt = (
        f"Given the Maha Shivaratri prompt: '{prompt}', generate two short, cinematic phrases. "
        f"1. An opening title (max 6 words). "
        f"2. A closing message (max 5 words). "
        f"Format as: Opening: [text] | Closing: [text]"
    )

    client = get_ai_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-nano-12b-v2-vl:free",
                messages=[{"role": "user", "content": chat_prompt}],
                max_tokens=300
            )
            text = response.choices[0].message.content
            if "|" in text:
                parts = text.split("|")
                op = parts[0].replace("Opening:", "").strip()
                cl = parts[1].replace("Closing:", "").strip()
                return op, cl
        except Exception as e:
            print(f"OpenRouter Error: {e}")

    # Fallback to direct Gemini
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(chat_prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=300))
            text = response.text
            if "|" in text:
                parts = text.split("|")
                op = parts[0].replace("Opening:", "").strip()
                cl = parts[1].replace("Closing:", "").strip()
                return op, cl
    except Exception as e:
        print(f"Gemini API Error: {e}")
    return None, None

def create_text_image(text, size=(1920, 1080), fontsize=70, color='white', bg_color=(0,0,0,0)):
    """Creates a text image using PIL to avoid ImageMagick dependency"""
    # Create transparent or colored background
    img = Image.new('RGBA', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Load default font
    try:
        # Try to load a generic font
        font = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position to center it
    # getbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    
    return np.array(img)

def create_promotional_video(theme='divine_blue', opening_text='The Night of Cosmic Awakening...', closing_text='Happy Maha Shivaratri', output_format='mp4', resolution='1080p', prompt='', output_path=None):
    # Use Gemini to refine text if prompt is provided
    if prompt:
        ai_op, ai_cl = get_gemini_vision(prompt)
        if ai_op: 
            opening_text = ai_op
            print(f"AI Opening Text: {opening_text}")
        if ai_cl: 
            closing_text = ai_cl
            print(f"AI Closing Text: {closing_text}")

    print(f"Generating Video | Theme: {theme} | Format: {output_format} | Resolution: {resolution} | AI Prompt: {prompt}")

    # Resolution Mapping
    res_map = {
        '360p': (640, 360),
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160)
    }
    video_size = res_map.get(resolution, (1920, 1080))
    
    # Font Scaling Factor
    scale = video_size[0] / 1920.0
    
    fps = 24
    
    # Theme Settings
    colors = {
        'divine_blue': {
            'scene1': (10, 10, 30),
            'scene2': (20, 5, 0),
            'scene5': (50, 0, 0)
        },
        'fiery_tandava': {
            'scene1': (30, 0, 0),
            'scene2': (40, 10, 0),
            'scene5': (80, 20, 0)
        },
        'golden_morning': {
            'scene1': (50, 40, 10),
            'scene2': (60, 50, 20),
            'scene5': (70, 30, 10)
        }
    }
    
    theme_colors = colors.get(theme, colors['divine_blue'])

    # Helper to make text clip
    def makeup_text_clip(txt, duration, fontsize=70, color='white'):
        img_array = create_text_image(txt, size=video_size, fontsize=fontsize, color=color)
        return ImageClip(img_array).set_duration(duration)

    # --- SCENE 1: Opening (5 sec) ---
    scene1_bg = ColorClip(size=video_size, color=theme_colors['scene1'], duration=5)
    scene1_text = makeup_text_clip(opening_text, 5, fontsize=int(70 * scale), color='white')
    scene1 = CompositeVideoClip([scene1_bg, scene1_text])
    
    # --- SCENE 2: Temple (8 sec) ---
    scene2_bg = ColorClip(size=video_size, color=theme_colors['scene2'], duration=8)
    scene2_text = makeup_text_clip("Himalayan Temple & Glowing Diyas", 8, fontsize=int(60 * scale), color='gold')
    scene2 = CompositeVideoClip([scene2_bg, scene2_text])

    # --- SCENE 3: Abhishekam (10 sec) ---
    scene3_bg = ColorClip(size=video_size, color=(200, 200, 220), duration=10)
    scene3_text = makeup_text_clip("Abhishekam & Om Namah Shivaya", 10, fontsize=int(60 * scale), color='black')
    scene3 = CompositeVideoClip([scene3_bg, scene3_text])

    # --- SCENE 4: Lord Shiva Darshan (10 sec) ---
    scene4_bg = ColorClip(size=video_size, color=(0, 20, 60), duration=10)
    scene4_text = makeup_text_clip("Lord Shiva Darshan - Divine Blue Aura", 10, fontsize=int(60 * scale), color='cyan')
    scene4 = CompositeVideoClip([scene4_bg, scene4_text])

    # --- SCENE 5: Tandava (10 sec) ---
    scene5_bg = ColorClip(size=video_size, color=theme_colors['scene5'], duration=10)
    scene5_text = makeup_text_clip("Tandava - Cosmic Dance", 10, fontsize=int(80 * scale), color='orange')
    scene5 = CompositeVideoClip([scene5_bg, scene5_text])

    # --- SCENE 6: Ending (7 sec) ---
    scene6_bg = ColorClip(size=video_size, color=(255, 223, 0), duration=7)
    full_closing_text = closing_text + "\n\nPowered by Google Gemini"
    scene6_text = makeup_text_clip(full_closing_text, 7, fontsize=int(60 * scale), color='black')
    scene6 = CompositeVideoClip([scene6_bg, scene6_text])

    # --- ASSEMBLE ---
    final_video = concatenate_videoclips([scene1, scene2, scene3, scene4, scene5, scene6])
    
    # Codec Settings
    if not output_path:
        output_path = f"maha_shivaratri_concept.{output_format}"
    
    codec = 'libx264'
    audio_codec = 'aac'

    if output_format == 'webm':
        codec = 'libvpx'
        audio_codec = 'libvorbis'
    elif output_format == 'avi':
        codec = 'mpeg4'
        audio_codec = 'mp3'

    final_video.write_videofile(output_path, fps=fps, codec=codec, audio_codec=audio_codec)
    print(f"Video saved as {output_path}")

if __name__ == "__main__":
    create_promotional_video()
