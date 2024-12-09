from .GRSubjects import subjects
from .GRMoods import moods
from .GRColors import colors
from .GRCompositions import compositions
from .GRDetails import details
from .GRWeather import weather
from .GRTime_of_day import time_of_day
from .GRObjects import objects
from .GRStyles import styles
import random

class HyperComplexImageDescriptionGenerator:
    def __init__(self):
        self.subjects = subjects
        self.moods = moods
        self.colors = colors
        self.compositions = compositions
        self.details = details
        self.weather = weather
        self.time_of_day = time_of_day
        self.objects = objects
        self.styles = styles

        self.types_of_images = ["random"] + sorted([
            "3D Render", "Anime-style", "Black and White", "Cartoon",
            "Charcoal Drawing", "Chiaroscuro", "Cinematic", "Claymation",
            "Concept Art", "Cubist", "Cyberpunk", "Doodle", "Double Exposure",
            "Embossed", "Engraving", "Etching", "Expressionist", "Fantasy",
            "Flat Design", "Glitch Art", "Gothic", "Grainy", "Grunge",
            "Hand-drawn", "High Contrast", "Holographic", "Hyper-realistic",
            "Illustrative", "Impressionistic", "Infrared", "Ink Drawing",
            "Isometric", "Low Poly", "Macro", "Metallic", "Minimalist",
            "Neon-lit", "Oil Painting", "Old", "Panoramic", "Papercut",
            "Pastel Drawing", "Photographic", "Pixel Art", "Pointillism",
            "Pop Art", "Realistic", "Renaissance", "Retro-futuristic",
            "Sepia", "Sketch", "Soft Focus", "Stained Glass", "Steampunk",
            "Stylized", "Surreal", "Synthwave", "Textured Collage",
            "Vaporwave", "Vintage", "Watercolor", "Woodcut", "Horror", "Scary", "Jumpy"
        ], key=str.casefold)

    def generate_prompt(
        self,
        seed,
        subject_type,
        category=None,
        replacement=None,
        image_type=None,
        mood_type=None,
        color_type=None,
        composition_type=None,
        detail_type=None,
        weather_type=None,
        time_of_day_type=None,
        object_type=None,
        style_type=None,
        subject_only=False,
        short_text=None,
    ):
        random.seed(seed)

        # Randomize image_type if it's "random"
        if image_type == "random" or not image_type:
                image_type = random.choice(self.types_of_images[1:])

        # Handle subject_only logic
        if subject_only:
                if short_text:
                        return f"This is a {image_type} image. {short_text}"
                if subject_type == "random":
                        subject_type = random.choice(list(self.subjects.keys()))
                subject = random.choice(self.subjects[subject_type]) if subject_type in self.subjects else "an undefined scene"
                return f"This is a {image_type} image. It captures {subject}."

        # Normal subject selection logic
        if subject_type == "random":
                subject_type = random.choice(list(self.subjects.keys()))
        subject = random.choice(self.subjects[subject_type]) if subject_type in self.subjects else "an undefined scene"

        if category == "subjects" and replacement:
                subject = replacement

        # Select attributes dynamically
        mood_options = self.moods.get(mood_type, list(self.moods.values())[0])  # Default to first category
        mood = None if mood_type == "none" else (replacement if category == "moods" else random.choice(mood_options))

        color_options = self.colors.get(color_type, list(self.colors.values())[0])  # Default to first category
        color = None if color_type == "none" else (replacement if category == "colors" else random.choice(color_options))

        composition_options = self.compositions.get(composition_type, list(self.compositions.values())[0])  # Default to first category
        composition = None if composition_type == "none" else (replacement if category == "compositions" else random.choice(composition_options))

        detail_options = self.details.get(detail_type, list(self.details.values())[0])  # Default to first category
        detail = None if detail_type == "none" else (replacement if category == "details" else random.choice(detail_options))

        weather_options = self.weather.get(weather_type, list(self.weather.values())[0])  # Default to first category
        weather = None if weather_type == "none" else (replacement if category == "weather" else random.choice(weather_options))

        time_of_day_options = self.time_of_day.get(time_of_day_type, list(self.time_of_day.values())[0])  # Default to first category
        time_of_day = None if time_of_day_type == "none" else (replacement if category == "time_of_day" else random.choice(time_of_day_options))

        object_options = self.objects.get(object_type, list(self.objects.values())[0])  # Default to first category
        obj = None if object_type == "none" else (replacement if category == "objects" else random.choice(object_options))

        style_options = self.styles.get(style_type, list(self.styles.values())[0])  # Default to first category
        style = None if style_type == "none" else (replacement if category == "styles" else random.choice(style_options))

        # Construct the prompt
        prompt = f"This is a {image_type} image, It captures {subject}"
        if mood:
                prompt += f" The mood is {mood}"
        if color:
                prompt += f" complemented by {color}"
        if composition:
                prompt += f" The composition features {composition}"
        if detail:
                prompt += f" enhanced by {detail}"
        if weather:
                prompt += f" The weather is described as {weather}"
        if time_of_day:
                prompt += f" and the time of day is {time_of_day}"
        if obj:
                prompt += f" In the foreground, {obj} stands out, drawing the eye"
        if style:
                prompt += f" The image is rendered in {style}"

        return prompt.strip()