import os
import sys
import re
import subprocess
import asyncio
import inspect
import shutil
import json
from datetime import datetime
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPalette, QPainter, QBrush, QPixmap, QFont, QIcon
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QSizePolicy
from googletrans import Translator
import whisper
import edge_tts
import yt_dlp

# -------- CONFIG --------
OUTPUT_FOLDER = os.path.join(os.getcwd(), "Alb")
PRESETS_FILE = os.path.join(os.getcwd(), "presets.json")
MUSIC_LIBRARY_FILE = os.path.join(os.getcwd(), "music_library.json")
WHISPER_MODEL = "base"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported platforms with example URLs
PLATFORMS = {
    "TikTok": {
        "name": "tiktok", 
        "regex": r"^(https?://)?(www\.)?tiktok\.com/.+/video/.+",
        "example": "https://www.tiktok.com/@username/video/1234567890"
    },
    "YouTube Shorts": {
        "name": "youtube", 
        "regex": r"^(https?://)?(www\.)?youtube\.com/shorts/.+",
        "example": "https://www.youtube.com/shorts/abc123def"
    },
    "Instagram": {
        "name": "instagram", 
        "regex": r"^(https?://)?(www\.)?instagram\.com/(p|reel)/.+",
        "example": "https://www.instagram.com/reel/ABC123DEF/"
    },
    "Facebook": {
        "name": "facebook", 
        "regex": r"^(https?://)?(www\.)?facebook\.com/.+/videos/.+",
        "example": "https://www.facebook.com/watch/?v=1234567890"
    }
}

# Supported languages + voices
LANGUAGES = {
    "Albanian": ("sq", ["sq-AL-IlirNeural", "sq-AL-AnilaNeural"]),
    "Croatian": ("hr", ["hr-HR-SreckoNeural", "hr-HR-JanaNeural"]),
    "Bosnian": ("bs", ["bs-BA-GoranNeural", "bs-BA-AmraNeural"]),
    "German": ("de", ["de-DE-KatjaNeural", "de-DE-ConradNeural"]),
    "Russian": ("ru", ["ru-RU-DariyaNeural", "ru-RU-PavelNeural"]),
    "French": ("fr", ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"]),
    "Chinese": ("zh-CN", ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"]),
    "Spanish": ("es", ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"]),
    "Italian": ("it", ["it-IT-ElsaNeural", "it-IT-LucaNeural"]),
    "English": ("en", ["en-US-JennyNeural", "en-US-GuyNeural"])
}

# Video quality presets
QUALITY_PRESETS = {
    "High (1080p)": {"height": 1080, "bitrate": "8000k"},
    "Medium (720p)": {"height": 720, "bitrate": "5000k"},
    "Low (480p)": {"height": 480, "bitrate": "2500k"},
    "Original": {"height": None, "bitrate": None}
}

# Watermark positions
WATERMARK_POSITIONS = {
    "Top Left": "10:10",
    "Top Center": "(w-text_w)/2:10",
    "Top Right": "w-text_w-10:10",
    "Center": "(w-text_w)/2:(h-text_h)/2",
    "Bottom Left": "10:h-text_h-10",
    "Bottom Center": "(w-text_w)/2:h-text_h-10",
    "Bottom Right": "w-text_w-10:h-text_h-10"
}

# -------- utilities --------
def run_ffmpeg_command(cmd: str) -> bool:
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=300)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("FFmpeg command timed out")
        return False

def master_audio(input_audio_path: str, output_audio_path: str, normalize: bool = True) -> bool:
    if normalize:
        cmd = f'ffmpeg -y -i "{input_audio_path}" -af "loudnorm=I=-16:TP=-1.5:LRA=11, highpass=f=200" "{output_audio_path}"'
    else:
        cmd = f'ffmpeg -y -i "{input_audio_path}" -af "highpass=f=200" "{output_audio_path}"'
    return run_ffmpeg_command(cmd)

def download_video(url: str, outpath: str, quality_preset: dict, platform: str):
    ydl_opts = {
        "outtmpl": outpath,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    
    # For YouTube Shorts, always get the best quality available
    if platform == "youtube":
        ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    # For other platforms, use quality settings
    elif quality_preset["height"]:
        ydl_opts["format"] = f"bestvideo[height<={quality_preset['height']}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
    except Exception as e:
        print(f"Download error: {e}")
        return None

def detect_platform(url: str) -> str:
    """Detect which platform the URL belongs to"""
    for platform, data in PLATFORMS.items():
        if re.match(data["regex"], url, re.IGNORECASE):
            return data["name"]
    return "tiktok"  # default to TikTok

# -------- Worker --------
class Worker(QThread):
    log_signal = pyqtSignal(str, str)  # message, type
    progress_signal = pyqtSignal(int, str)  # percent, text
    finished_signal = pyqtSignal(int, int)  # success_count, total

    def __init__(self, urls, language_code, voice_name, bg_volume=30, output_folder=OUTPUT_FOLDER, 
                 watermark_text="", quality_preset=None, generate_subtitles=False,
                 audio_normalize=True, retry_attempts=2, platform="tiktok", 
                 watermark_position="Bottom Right"):
        super().__init__()
        self.urls = urls
        self.language_code = language_code
        self.voice_name = voice_name
        self.bg_volume = bg_volume
        self.output_folder = output_folder
        self.watermark_text = watermark_text
        self.quality_preset = quality_preset or QUALITY_PRESETS["Original"]
        self.generate_subtitles = generate_subtitles
        self.audio_normalize = audio_normalize
        self.retry_attempts = retry_attempts
        self.platform = platform
        self.watermark_position = watermark_position
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.log_signal.emit(f"Worker crashed: {e}", "error")
            self.finished_signal.emit(0, len(self.urls))

    async def _main(self):
        success = 0
        total = len(self.urls)
        temp_dir = os.path.join(self.output_folder, ".temp")
        os.makedirs(temp_dir, exist_ok=True)

        self.log_signal.emit("Loading Whisper model...", "info")
        try:
            # Check if GPU is available for Whisper
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Using device: {device}", "info")
            model = whisper.load_model(WHISPER_MODEL, device=device)
        except Exception as e:
            self.log_signal.emit(f"Failed loading Whisper model: {e}", "error")
            self.finished_signal.emit(0, total)
            return

        translator = Translator()
        for idx, url in enumerate(self.urls, start=1):
            if self._stop:
                self.log_signal.emit("Stopped by user.", "warning")
                break
            
            self.progress_signal.emit(int((idx-1)/total*100), f"Processing ({idx}/{total}): {url}")
            self.log_signal.emit(f"Processing ({idx}/{total}): {url}", "info")

            # Retry logic
            for attempt in range(self.retry_attempts + 1):
                try:
                    ok = await self._process_single(url, temp_dir, model, translator)
                    if ok:
                        success += 1
                        break
                    elif attempt < self.retry_attempts:
                        self.log_signal.emit(f"Retrying ({attempt + 1}/{self.retry_attempts})...", "warning")
                except Exception as e:
                    if attempt < self.retry_attempts:
                        self.log_signal.emit(f"Retrying after error: {e}", "warning")
                    else:
                        self.log_signal.emit(f"Failed after {self.retry_attempts} attempts: {e}", "error")

        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        
        self.progress_signal.emit(100, "Processing complete!")
        self.finished_signal.emit(success, total)

    async def _process_single(self, url, temp_dir, model, translator) -> bool:
        video_id_match = re.search(r"/video/(\d+)", url) or re.search(r"shorts/([^/?]+)", url) or re.search(r"reel/([^/?]+)", url)
        filename = f"{self.platform}_{video_id_match.group(1) if video_id_match else datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        temp_video_path = os.path.join(temp_dir, filename)
        final_output_path = os.path.join(self.output_folder, f"{self.language_code}_{filename}")
        subtitle_path = os.path.splitext(final_output_path)[0] + ".srt"

        if os.path.exists(final_output_path):
            self.log_signal.emit(f"Already exists: {final_output_path}", "info")
            return True

        self.log_signal.emit("Downloading video...", "info")
        downloaded = download_video(url, temp_video_path, self.quality_preset, self.platform)
        if not downloaded or not os.path.exists(downloaded):
            self.log_signal.emit("Download failed, skipping.", "error")
            return False

        base = os.path.splitext(filename)[0]
        extracted_audio = os.path.join(temp_dir, f"{base}_orig.wav")
        music_audio = os.path.join(temp_dir, f"{base}_music.wav")
        translated_audio = os.path.join(temp_dir, f"{base}_{self.language_code}.mp3")
        mastered_audio = os.path.join(temp_dir, f"{base}_{self.language_code}_master.mp3")
        video_no_audio = os.path.join(temp_dir, f"{base}_noaudio.mp4")
        final_with_watermark = os.path.join(temp_dir, f"{base}_watermark.mp4")

        try:
            # Extract full audio
            self.log_signal.emit("Extracting audio...", "info")
            cmd_extract = f'ffmpeg -y -i "{downloaded}" -vn -q:a 0 -map a "{extracted_audio}"'
            if not run_ffmpeg_command(cmd_extract):
                self.log_signal.emit("Audio extraction failed.", "error")
                return False

            # Separate background music (rough vocal removal)
            self.log_signal.emit("Extracting background music...", "info")
            cmd_music = f'ffmpeg -y -i "{extracted_audio}" -af "highpass=f=200, lowpass=f=3000, volume={self.bg_volume/100}" "{music_audio}"'
            run_ffmpeg_command(cmd_music)

            # Transcribe
            self.log_signal.emit("Transcribing...", "info")
            try:
                result = model.transcribe(extracted_audio, language="en")
                english_text = result.get("text", "").strip()
                
                # Get timing information for text-to-speech alignment
                segments = result.get("segments", [])
            except Exception as e:
                self.log_signal.emit(f"Transcription failed: {e}", "error")
                return False

            if not english_text:
                self.log_signal.emit("No speech detected.", "error")
                return False

            # Generate subtitles if requested
            if self.generate_subtitles and segments:
                self.log_signal.emit("Generating subtitles...", "info")
                try:
                    with open(subtitle_path, 'w', encoding='utf-8') as srt:
                        for i, segment in enumerate(segments):
                            start = segment['start']
                            end = segment['end']
                            text = segment['text']
                            
                            # Convert to SRT time format
                            start_time = datetime.utcfromtimestamp(start).strftime('%H:%M:%S,%f')[:-3]
                            end_time = datetime.utcfromtimestamp(end).strftime('%H:%M:%S,%f')[:-3]
                            
                            srt.write(f"{i+1}\n")
                            srt.write(f"{start_time} --> {end_time}\n")
                            srt.write(f"{text}\n\n")
                except Exception as e:
                    self.log_signal.emit(f"Subtitle generation failed: {e}", "warning")

            # Translate
            self.log_signal.emit(f"Translating to {self.language_code}...", "info")
            try:
                maybe = translator.translate(english_text, src="en", dest=self.language_code)
                if asyncio.iscoroutine(maybe) or inspect.isawaitable(maybe):
                    translation = await maybe
                else:
                    translation = maybe
                translated_text = getattr(translation, "text", None)
                if not translated_text:
                    self.log_signal.emit("Translation returned empty text.", "error")
                    return False
            except Exception as e:
                self.log_signal.emit(f"Translation failed: {e}", "error")
                return False

            # TTS with timing adjustment
            self.log_signal.emit("Generating speech (TTS)...", "info")
            try:
                communicate = edge_tts.Communicate(translated_text, self.voice_name)
                await communicate.save(translated_audio)
            except Exception as e:
                self.log_signal.emit(f"TTS generation failed: {e}", "error")
                return False

            # Master audio
            self.log_signal.emit("Mastering audio...", "info")
            if not master_audio(translated_audio, mastered_audio, self.audio_normalize):
                shutil.copyfile(translated_audio, mastered_audio)

            # Strip original audio from video
            self.log_signal.emit("Removing original audio...", "info")
            cmd_strip = f'ffmpeg -y -i "{downloaded}" -c:v copy -an "{video_no_audio}"'
            if not run_ffmpeg_command(cmd_strip):
                self.log_signal.emit("Failed stripping original audio.", "error")
                return False

            # Merge final: TTS + background music - FIXED VERSION
            self.log_signal.emit("Merging new audio + background music...", "info")
            
            # First, mix the audio tracks
            mixed_audio = os.path.join(temp_dir, f"{base}_mixed_audio.aac")
            mix_cmd = (
                f'ffmpeg -y -i "{mastered_audio}" -i "{music_audio}" '
                f'-filter_complex "[0:a]volume=1.0[a1];[1:a]volume={self.bg_volume/100}[a2];[a1][a2]amix=inputs=2:duration=longest" '
                f'-c:a aac -b:a 192k "{mixed_audio}"'
            )
            
            if not run_ffmpeg_command(mix_cmd):
                self.log_signal.emit("Failed mixing audio tracks.", "error")
                return False
            
            # Then merge the mixed audio with the video
            # Add quality settings if specified
            quality_args = ""
            if self.quality_preset["bitrate"]:
                quality_args = f'-b:v {self.quality_preset["bitrate"]} -maxrate {self.quality_preset["bitrate"]} -bufsize 2M'
            
            merge_cmd = (
                f'ffmpeg -y -i "{video_no_audio}" -i "{mixed_audio}" '
                f'-c:v copy {quality_args} -c:a copy -shortest "{final_output_path}"'
            )
            
            if not run_ffmpeg_command(merge_cmd):
                self.log_signal.emit("Failed merging audio and video.", "error")
                return False

            # Add watermark if specified
            if self.watermark_text:
                self.log_signal.emit("Adding watermark...", "info")
                # Get position coordinates
                position = WATERMARK_POSITIONS.get(self.watermark_position, "w-text_w-10:h-text_h-10")
                
                # Use drawtext filter with high quality settings
                # Escape special characters in watermark text
                escaped_text = self.watermark_text.replace("'", "'\\''")
                
                watermark_cmd = (
                    f'ffmpeg -y -i "{final_output_path}" '
                    f'-vf "drawtext=text=\'{escaped_text}\':fontcolor=white:fontsize=24:'
                    f'box=1:boxcolor=black@0.5:boxborderw=5:'
                    f'x={position}" '
                    f'-c:a copy "{final_with_watermark}"'
                )
                if run_ffmpeg_command(watermark_cmd):
                    # Replace the final output with watermarked version
                    shutil.move(final_with_watermark, final_output_path)
                    self.log_signal.emit("Watermark added successfully", "success")
                else:
                    self.log_signal.emit("Watermark failed, using original video", "warning")

            self.log_signal.emit(f"Saved: {final_output_path}", "success")
            if self.generate_subtitles and os.path.exists(subtitle_path):
                self.log_signal.emit(f"Subtitles: {subtitle_path}", "success")
            return True

        except Exception as e:
            self.log_signal.emit(f"Error processing video: {str(e)}", "error")
            return False
        finally:
            for f in [downloaded, extracted_audio, music_audio, translated_audio, 
                     mastered_audio, video_no_audio, final_with_watermark,
                     os.path.join(temp_dir, f"{base}_mixed_audio.aac")]:
                try:
                    if f and os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass

# -------- Modern UI Components --------
class ModernButton(QtWidgets.QPushButton):
    def __init__(self, text, primary=False, parent=None):
        super().__init__(text, parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background: #8B5CF6;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: #7C3AED;
                }
                QPushButton:pressed {
                    background: #6D28D9;
                }
                QPushButton:disabled {
                    background: #6B7280;
                    color: #9CA3AF;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.08);
                    color: #E5E7EB;
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 8px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.12);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.05);
                }
                QPushButton:disabled {
                    background: rgba(255, 255, 255, 0.05);
                    color: #6B7280;
                }
            """)

class ModernCard(QtWidgets.QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(31, 41, 55, 0.7);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        if title:
            title_label = QtWidgets.QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #E5E7EB;
                    font-weight: bold;
                    font-size: 16px;
                }
            """)
            layout.addWidget(title_label)
        
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_layout.setSpacing(10)
        layout.addLayout(self.content_layout)

class ModernComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                background: rgba(255, 255, 255, 0.08);
                color: #E5E7EB;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QComboBox:focus {
                border: 1px solid #8B5CF6;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                background: rgba(31, 41, 55, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.12);
                selection-background-color: #8B5CF6;
                selection-color: white;
                color: #E5E7EB;
            }
        """)
        self.setMinimumHeight(36)

class ModernTextEdit(QtWidgets.QTextEdit):
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.08);
                color: #E5E7EB;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
            }
            QTextEdit:focus {
                border: 1px solid #8B5CF6;
            }
            QTextEdit::placeholder {
                color: #6B7280;
            }
        """)

class ModernLineEdit(QtWidgets.QLineEdit):
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.08);
                color: #E5E7EB;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #8B5CF6;
            }
            QLineEdit::placeholder {
                color: #6B7280;
            }
        """)
        self.setMinimumHeight(36)

class ModernSlider(QtWidgets.QSlider):
    def __init__(self, orientation=QtCore.Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 0.1);
                height: 5px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #8B5CF6;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #8B5CF6;
                border-radius: 2px;
            }
        """)

class ModernCheckBox(QtWidgets.QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                color: #E5E7EB;
                font-size: 13px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.08);
            }
            QCheckBox::indicator:checked {
                background: #8B5CF6;
                border: 1px solid #8B5CF6;
            }
            QCheckBox::indicator:checked:hover {
                background: #7C3AED;
                border: 1px solid #7C3AED;
            }
        """)

class ModernProgressBar(QtWidgets.QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #8B5CF6;
                border-radius: 4px;
            }
        """)
        self.setTextVisible(False)

# -------- Modern UI --------
class TikTokTranslatorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = {
            "output_folder": OUTPUT_FOLDER,
            "watermark_text": "",
            "quality_preset": QUALITY_PRESETS["Original"],
            "generate_subtitles": False,
            "audio_normalize": True,
            "retry_attempts": 2,
            "platform": "tiktok",
            "watermark_position": "Bottom Right"
        }
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Video Translator Pro")
        self.setMinimumSize(1000, 700)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background: #111827;
            }
            QLabel {
                color: #E5E7EB;
            }
        """)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # Left panel (content)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)
        
        # Header
        header = QtWidgets.QLabel("Video Translator Pro")
        header.setStyleSheet("""
            QLabel {
                color: #8B5CF6;
                font-size: 22px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        left_layout.addWidget(header)
        
        # Video URLs card
        url_card = ModernCard("Video Sources")
        
        platform_layout = QtWidgets.QHBoxLayout()
        platform_layout.addWidget(QtWidgets.QLabel("Platform:"))
        self.platform_combo = ModernComboBox()
        self.platform_combo.addItems(PLATFORMS.keys())
        self.platform_combo.currentTextChanged.connect(self.update_url_placeholder)
        platform_layout.addWidget(self.platform_combo)
        url_card.content_layout.addLayout(platform_layout)
        
        url_label = QtWidgets.QLabel("Video URLs (one per line):")
        url_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        url_card.content_layout.addWidget(url_label)
        
        self.url_input = ModernTextEdit()
        self.url_input.setFixedHeight(100)
        url_card.content_layout.addWidget(self.url_input)
        
        left_layout.addWidget(url_card)
        
        # Translation settings card
        trans_card = ModernCard("Translation Settings")
        
        lang_layout = QtWidgets.QHBoxLayout()
        lang_layout.addWidget(QtWidgets.QLabel("Language:"))
        self.lang_combo = ModernComboBox()
        self.lang_combo.addItems(LANGUAGES.keys())
        self.lang_combo.currentTextChanged.connect(self.update_voices)
        lang_layout.addWidget(self.lang_combo)
        trans_card.content_layout.addLayout(lang_layout)
        
        voice_layout = QtWidgets.QHBoxLayout()
        voice_layout.addWidget(QtWidgets.QLabel("Voice:"))
        self.voice_combo = ModernComboBox()
        voice_layout.addWidget(self.voice_combo)
        trans_card.content_layout.addLayout(voice_layout)
        
        # Volume slider
        vol_layout = QtWidgets.QVBoxLayout()
        vol_label_layout = QtWidgets.QHBoxLayout()
        vol_label_layout.addWidget(QtWidgets.QLabel("Background Music Volume:"))
        self.vol_label = QtWidgets.QLabel("30%")
        self.vol_label.setStyleSheet("color: #8B5CF6; font-weight: bold;")
        vol_label_layout.addWidget(self.vol_label)
        vol_layout.addLayout(vol_label_layout)
        
        self.vol_slider = ModernSlider()
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(30)
        self.vol_slider.valueChanged.connect(lambda v: self.vol_label.setText(f"{v}%"))
        vol_layout.addWidget(self.vol_slider)
        
        trans_card.content_layout.addLayout(vol_layout)
        left_layout.addWidget(trans_card)
        
        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.start_btn = ModernButton("Start Processing", primary=True)
        self.start_btn.clicked.connect(self.start_clicked)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = ModernButton("Stop")
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.open_folder_btn = ModernButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        left_layout.addLayout(btn_layout)
        
        # Progress bar
        progress_layout = QtWidgets.QVBoxLayout()
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_text = QtWidgets.QLabel("Ready to process videos")
        self.progress_text.setStyleSheet("color: #9CA3AF; text-align: center; font-size: 12px;")
        progress_layout.addWidget(self.progress_text)
        
        left_layout.addLayout(progress_layout)
        
        # Settings button
        settings_btn = ModernButton("Advanced Settings")
        settings_btn.clicked.connect(self.show_settings_dialog)
        left_layout.addWidget(settings_btn)
        
        left_layout.addStretch()
        
        # Right panel (log)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        log_card = ModernCard("Processing Log")
        log_card.setFixedWidth(350)
        
        self.log_display = QtWidgets.QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.05);
                color: #E5E7EB;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.log_display.setFixedHeight(500)
        log_card.content_layout.addWidget(self.log_display)
        
        # Clear log button
        clear_btn = ModernButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log_display.clear())
        log_card.content_layout.addWidget(clear_btn)
        
        right_layout.addWidget(log_card)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)
        
        # Initialize
        self.worker = None
        self.update_voices()
        self.update_url_placeholder()

    def update_voices(self):
        name = self.lang_combo.currentText()
        code, voices = LANGUAGES.get(name, ("en", ["en-US-JennyNeural"]))
        self.voice_combo.clear()
        self.voice_combo.addItems(voices)

    def update_url_placeholder(self):
        platform = self.platform_combo.currentText()
        example_url = PLATFORMS.get(platform, {}).get("example", "https://example.com/video/123")
        self.url_input.setPlaceholderText(f"Paste {platform} URLs (one per line)\nExample: {example_url}")

    def show_settings_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Advanced Settings")
        dialog.setMinimumWidth(500)
        dialog.setStyleSheet("""
            QDialog {
                background: #111827;
            }
            QLabel {
                color: #E5E7EB;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Output settings
        output_card = ModernCard("Output Settings")
        
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setSpacing(10)
        
        output_folder_layout = QtWidgets.QHBoxLayout()
        output_folder_layout.addWidget(QtWidgets.QLabel("Output Folder:"))
        output_edit = ModernLineEdit(self.settings["output_folder"])
        output_folder_layout.addWidget(output_edit)
        browse_btn = ModernButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_output_folder(output_edit))
        output_folder_layout.addWidget(browse_btn)
        output_layout.addLayout(output_folder_layout)
        
        quality_layout = QtWidgets.QHBoxLayout()
        quality_layout.addWidget(QtWidgets.QLabel("Video Quality:"))
        quality_combo = ModernComboBox()
        quality_combo.addItems(QUALITY_PRESETS.keys())
        quality_combo.setCurrentText([k for k, v in QUALITY_PRESETS.items() if v == self.settings["quality_preset"]][0])
        quality_layout.addWidget(quality_combo)
        output_layout.addLayout(quality_layout)
        
        output_card.content_layout.addLayout(output_layout)
        layout.addWidget(output_card)
        
        # Watermark settings
        watermark_card = ModernCard("Watermark Settings")
        
        watermark_layout = QtWidgets.QVBoxLayout()
        watermark_layout.setSpacing(10)
        
        watermark_text_layout = QtWidgets.QHBoxLayout()
        watermark_text_layout.addWidget(QtWidgets.QLabel("Watermark Text:"))
        watermark_edit = ModernLineEdit(self.settings["watermark_text"])
        watermark_text_layout.addWidget(watermark_edit)
        watermark_layout.addLayout(watermark_text_layout)
        
        watermark_pos_layout = QtWidgets.QHBoxLayout()
        watermark_pos_layout.addWidget(QtWidgets.QLabel("Watermark Position:"))
        watermark_pos_combo = ModernComboBox()
        watermark_pos_combo.addItems(WATERMARK_POSITIONS.keys())
        watermark_pos_combo.setCurrentText(self.settings["watermark_position"])
        watermark_pos_layout.addWidget(watermark_pos_combo)
        watermark_layout.addLayout(watermark_pos_layout)
        
        watermark_card.content_layout.addLayout(watermark_layout)
        layout.addWidget(watermark_card)
        
        # Advanced settings
        advanced_card = ModernCard("Advanced Settings")
        
        advanced_layout = QtWidgets.QVBoxLayout()
        advanced_layout.setSpacing(10)
        
        retry_layout = QtWidgets.QHBoxLayout()
        retry_layout.addWidget(QtWidgets.QLabel("Retry Attempts:"))
        retry_spin = QtWidgets.QSpinBox()
        retry_spin.setRange(0, 5)
        retry_spin.setValue(self.settings["retry_attempts"])
        retry_layout.addWidget(retry_spin)
        advanced_layout.addLayout(retry_layout)
        
        subtitle_check = ModernCheckBox("Generate Subtitles (SRT)")
        subtitle_check.setChecked(self.settings["generate_subtitles"])
        advanced_layout.addWidget(subtitle_check)
        
        normalize_check = ModernCheckBox("Normalize Audio")
        normalize_check.setChecked(self.settings["audio_normalize"])
        advanced_layout.addWidget(normalize_check)
        
        advanced_card.content_layout.addLayout(advanced_layout)
        layout.addWidget(advanced_card)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        save_btn = ModernButton("Save", primary=True)
        save_btn.clicked.connect(lambda: self.save_settings(
            output_edit.text(),
            QUALITY_PRESETS[quality_combo.currentText()],
            watermark_edit.text(),
            watermark_pos_combo.currentText(),
            retry_spin.value(),
            subtitle_check.isChecked(),
            normalize_check.isChecked(),
            dialog
        ))
        button_layout.addWidget(save_btn)
        
        cancel_btn = ModernButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def save_settings(self, output_folder, quality_preset, watermark_text, watermark_position, 
                     retry_attempts, generate_subtitles, audio_normalize, dialog):
        self.settings.update({
            "output_folder": output_folder,
            "quality_preset": quality_preset,
            "watermark_text": watermark_text,
            "watermark_position": watermark_position,
            "retry_attempts": retry_attempts,
            "generate_subtitles": generate_subtitles,
            "audio_normalize": audio_normalize
        })
        dialog.accept()
        self.append_log("Settings saved successfully", "success")

    def browse_output_folder(self, output_edit):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", OUTPUT_FOLDER)
        if folder:
            output_edit.setText(folder)

    def append_log(self, message, log_type="info"):
        color_map = {
            "info": "#60A5FA",    # Blue
            "success": "#34D399", # Green
            "warning": "#FBBF24", # Yellow
            "error": "#F87171"    # Red
        }
        
        color = color_map.get(log_type, "#e6e6e6")
        timestamp = datetime.now().strftime("%H:%M:%S")
        html = f'<font color="{color}">[{timestamp}] {message}</font><br>'
        
        # Save scroll position
        scrollbar = self.log_display.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        
        # Append message
        self.log_display.append(html)
        
        # Scroll to bottom if was at bottom
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def start_clicked(self):
        raw = self.url_input.toPlainText().strip()
        urls = [u.strip() for u in raw.splitlines() if u.strip()]
        if not urls:
            self.append_log("⚠️ Paste at least one video URL.", "warning")
            return
        
        # Update platform setting
        self.settings["platform"] = PLATFORMS[self.platform_combo.currentText()]["name"]
        
        lang_name = self.lang_combo.currentText()
        lang_code = LANGUAGES.get(lang_name, ("en", []))[0]
        voice = self.voice_combo.currentText() or LANGUAGES.get(lang_name, ("en", []))[1][0]
        bg_volume = self.vol_slider.value()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.lang_combo.setEnabled(False)
        self.voice_combo.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_text.setText("Starting...")

        self.worker = Worker(
            urls, lang_code, voice, bg_volume, 
            self.settings["output_folder"], 
            self.settings["watermark_text"],
            self.settings["quality_preset"],
            self.settings["generate_subtitles"],
            self.settings["audio_normalize"],
            self.settings["retry_attempts"],
            self.settings["platform"],
            self.settings["watermark_position"]
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.append_log(f"🚀 Starting batch of {len(urls)} videos (language={lang_code}, voice={voice}, bg_volume={bg_volume}%)", "info")
        self.worker.start()

    def update_progress(self, percent, text):
        self.progress_bar.setValue(percent)
        self.progress_text.setText(text)

    def stop_clicked(self):
        if self.worker:
            self.append_log("Stopping worker (will stop after current video)...", "warning")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def on_finished(self, success_count, total):
        self.append_log(f"🎉 Batch finished! {success_count}/{total} videos translated successfully.", "success")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.lang_combo.setEnabled(True)
        self.voice_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_text.setText("Ready to process videos")

    def open_output_folder(self):
        output_path = self.settings["output_folder"]
        if os.path.exists(output_path):
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_path])
            else:
                subprocess.Popen(["xdg-open", output_path])
        else:
            self.append_log("Output folder does not exist", "error")

# -------- run --------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style for modern look
    app.setStyle("Fusion")
    
    # Set dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(17, 24, 39))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(31, 41, 55))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(17, 24, 39))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(17, 24, 39))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(139, 92, 246))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    window = TikTokTranslatorApp()
    window.show()
    sys.exit(app.exec())