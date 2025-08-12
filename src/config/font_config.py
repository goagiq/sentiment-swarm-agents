"""
Font Configuration for Multilingual Support
Handles language-specific font settings for matplotlib visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List
import platform


class FontConfig:
    """Configuration for language-specific font settings."""
    
    def __init__(self):
        self.font_mappings = self._get_font_mappings()
        self.fallback_fonts = self._get_fallback_fonts()
        self._cache_available_fonts()
    
    def _get_font_mappings(self) -> Dict[str, List[str]]:
        """Get font mappings for different languages."""
        system = platform.system().lower()
        
        if system == "windows":
            return {
                "zh": [
                    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
                    "Arial Unicode MS", "Segoe UI", "Tahoma", "Verdana"
                ],
                "ja": [
                    "Yu Gothic", "Meiryo", "MS Gothic", "MS Mincho",
                    "Arial Unicode MS", "Segoe UI"
                ],
                "ko": [
                    "Malgun Gothic", "Batang", "Dotum", "Gulim",
                    "Arial Unicode MS", "Segoe UI"
                ],
                "ru": [
                    "Times New Roman", "Arial", "Calibri", "Segoe UI",
                    "Arial Unicode MS", "Verdana"
                ],
                "en": [
                    "Arial", "Helvetica", "Times New Roman", "Calibri",
                    "Segoe UI", "Verdana", "DejaVu Sans"
                ],
                "default": [
                    "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"
                ]
            }
        elif system == "darwin":  # macOS
            return {
                "zh": [
                    "PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS",
                    "Helvetica Neue", "Helvetica"
                ],
                "ja": [
                    "Hiragino Kaku Gothic ProN", "Hiragino Sans", "Yu Gothic",
                    "Arial Unicode MS", "Helvetica Neue"
                ],
                "ko": [
                    "Apple SD Gothic Neo", "Arial Unicode MS", "Helvetica Neue"
                ],
                "ru": [
                    "Times New Roman", "Arial", "Helvetica Neue", "Arial Unicode MS"
                ],
                "en": [
                    "Helvetica Neue", "Helvetica", "Arial", "Times New Roman",
                    "Arial Unicode MS"
                ],
                "default": [
                    "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"
                ]
            }
        else:  # Linux and others
            return {
                "zh": [
                    "Noto Sans CJK SC", "Noto Sans CJK TC", "WenQuanYi Micro Hei",
                    "Droid Sans Fallback", "DejaVu Sans"
                ],
                "ja": [
                    "Noto Sans CJK JP", "Takao", "VL Gothic", "DejaVu Sans"
                ],
                "ko": [
                    "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"
                ],
                "ru": [
                    "Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"
                ],
                "en": [
                    "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"
                ],
                "default": [
                    "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"
                ]
            }
    
    def _get_fallback_fonts(self) -> List[str]:
        """Get fallback fonts that should work on most systems."""
        return [
            "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica",
            "Times New Roman", "Verdana", "Tahoma"
        ]
    
    def _cache_available_fonts(self):
        """Cache available fonts for faster lookup."""
        self._available_fonts = set()
        try:
            for font in fm.findSystemFonts():
                try:
                    font_prop = fm.FontProperties(fname=font)
                    self._available_fonts.add(font_prop.get_name())
                except Exception:
                    continue
        except Exception as e:
            print(f"Warning: Could not cache fonts: {e}")
    
    def _is_font_available(self, font_name: str) -> bool:
        """Check if a font is available on the system."""
        try:
            # First check our cache
            if hasattr(self, '_available_fonts') and font_name in self._available_fonts:
                return True
            
            # Try to find the font
            font_path = fm.findfont(
                fm.FontProperties(family=font_name), 
                fallback_to_default=False
            )
            return font_path != fm.rcParams['font.sans-serif'][0]
        except Exception:
            return False
    
    def get_font_family(self, language: str) -> str:
        """Get the best available font family for a given language."""
        # Normalize language code
        lang_code = language.lower().split('-')[0]
        
        # Get font options for this language
        font_options = self.font_mappings.get(lang_code, self.font_mappings["default"])
        
        # Try each font option
        for font in font_options:
            if self._is_font_available(font):
                return font
        
        # If no language-specific font found, try fallbacks
        for font in self.fallback_fonts:
            if self._is_font_available(font):
                return font
        
        # Last resort - return the first available font
        return "DejaVu Sans"
    
    def configure_font_for_language(self, language: str) -> bool:
        """Configure matplotlib font for a specific language."""
        try:
            font_family = self.get_font_family(language)
            
            # Configure matplotlib
            plt.rcParams['font.family'] = font_family
            plt.rcParams['axes.unicode_minus'] = False
            
            # For Chinese, Japanese, Korean, ensure proper Unicode handling
            if language.lower().startswith(('zh', 'ja', 'ko')):
                plt.rcParams['font.sans-serif'] = [font_family] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            
            return True
        except Exception as e:
            print(f"Warning: Could not configure font for {language}: {e}")
            return False
    
    def test_font_rendering(self, language: str, test_text: str = "测试") -> bool:
        """Test if a font can render text properly."""
        try:
            font_family = self.get_font_family(language)
            
            # Create a simple test plot
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, test_text, fontsize=20, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Try to render
            fig.canvas.draw()
            plt.close(fig)
            return True
        except Exception as e:
            print(f"Font rendering test failed for {language}: {e}")
            return False


# Global font configuration instance
_font_config = FontConfig()


def get_font_family(language: str) -> str:
    """Get the best available font family for a given language."""
    return _font_config.get_font_family(language)


def configure_font_for_language(language: str) -> bool:
    """Configure matplotlib font for a specific language."""
    return _font_config.configure_font_for_language(language)


def test_font_rendering(language: str, test_text: str = None) -> bool:
    """Test if a font can render text properly."""
    if test_text is None:
        # Default test texts for different languages
        test_texts = {
            "zh": "测试文字",
            "ja": "テスト文字",
            "ko": "테스트 텍스트",
            "ru": "Тестовый текст",
            "en": "Test Text"
        }
        test_text = test_texts.get(language.lower().split('-')[0], "Test")
    
    return _font_config.test_font_rendering(language, test_text)


def install_system_fonts():
    """Attempt to install system fonts if needed."""
    system = platform.system().lower()
    
    if system == "windows":
        print("On Windows, please ensure Chinese fonts are installed:")
        print("- Microsoft YaHei (usually pre-installed)")
        print("- SimHei (usually pre-installed)")
        print("- Or install fonts from Windows Settings > Personalization > Fonts")
    
    elif system == "darwin":  # macOS
        print("On macOS, Chinese fonts should be available by default:")
        print("- PingFang SC")
        print("- Hiragino Sans GB")
    
    else:  # Linux
        print("On Linux, you may need to install fonts:")
        print("sudo apt-get install fonts-noto-cjk  # Ubuntu/Debian")
        print("sudo yum install google-noto-cjk-fonts  # CentOS/RHEL")


def get_font_status_report():
    """Generate a report of font availability for different languages."""
    languages = ["zh", "ja", "ko", "ru", "en"]
    report = {}
    
    for lang in languages:
        font_family = get_font_family(lang)
        can_render = test_font_rendering(lang)
        report[lang] = {
            "font_family": font_family,
            "can_render": can_render,
            "available": _font_config._is_font_available(font_family)
        }
    
    return report
