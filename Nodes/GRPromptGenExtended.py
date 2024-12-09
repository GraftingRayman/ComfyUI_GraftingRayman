from .GRDescriptionDB import HyperComplexImageDescriptionGenerator
import random

class GRPromptGenExtended:
    _categories = [
        "subjects", "moods", "colors", "compositions", "details",
        "weather", "time_of_day", "objects", "styles"
    ]
    _mood_types = ["none","random"] + sorted(["positive", "negative", "mixed", "fear", "energetic", "low_energy", "atmospheres", "events", "adventures", "energies", "introspections", "textures", "tones", "mysteries", "forces"])
    _color_types = ["none","random"] + sorted(["warm", "cool", "neutral", "vivid", "earthy", "pastel", "metallic", "luxurious", "shadowy", "underwater"])
    _composition_types = ["none","random"] + sorted(["underwater", "deep_dark", "symmetry", "flow", "contrast", "perspective", "dynamic", "pattern", "classical", "modern", "experimental", "framing", "light", "juxtaposition", "textual", "radial", "layered", "narrative", "abstract", "rhythmic", "minimalist", "surreal", "geometric", "organic", "horror", "focus"])
    _detail_types = ["none","random"] + sorted(["water", "light", "texture","movement", "atmosphere", "nature", "sound", "macabre","underwater", "deep_dark"])
    _weather_types = ["none","random"] + sorted(["clear", "precipitation", "extreme", "rainy", "sunny", "snowy", "windy", "foggy", "stormy", "cold", "hot", "dramatic", "ethereal", "seasonal", "tropical", "mystical", "oceanic", "desert", "mountainous", "urban", "romantic", "wild", "nocturnal", "frosty", "energetic", "heavy", "ephermeral", "tranquil", "intense", "dull", "radiant", "isolated", "ominous", "hazy", "auroral", "blustery", "meditative", "electrifying", "refreshing", "overbearing", "chaotic", "bleak", "comforting", "distant", "rugged", "transient", "sublime", "oppresive", "exuberant", "idyllic", "subtle", "enveloping", "polar", "romanticized", "calm", "misty", "frightening"])
    _time_of_day_types = ["none","random"] + sorted(["daylight", "evening", "night", "morning", "midday", "afternoon", "dusk", "golden hour", "twilight", "dawn", "midnight", "sunrise", "sunset", "pre-dawn", "early morning", "late morning", "early afternoon", "late evening", "early dusk", "deep night", "early twilight", "early golden hour","noon", "late night", "early midnight", "late midnight", "early sunrise", "late sunrise", "early sunset", "late sunset"])
    _object_types = ["none","random"] + sorted(["natural", "manmade", "futuristic", "signage", "nature", "water", "buildings", "objects", "gathering", "structures", "mechanisms", "decor", "tools", "artifacts", "transportation", "landscape", "lighting", "furniture", "festivities", "flora", "fauna", "mystery", "weather", "relics", "navigation", "craftmanship", "play", "symbols", "storage", "wearables", "seasons", "gardening", "adornment", "abandonment", "celebration", "craft", "heritage", "energy", "industry", "exploration", "agriculture", "storytelling", "coziness", "marine", "fragility", "mobility", "solitude", "trade", "community", "tradition", "balance", "imagination", "connections", "evanescence", "ominous", "underwater"])
    _style_types = ["none","random"] + sorted(["classic", "modern", "digital", "hyper-realistic", "impressionistic", "surreal", "minimalist", "painterly", "dreamlike", "noir", "vintage", "whimsical", "futuristic", "fantasy", "abstract", "retro", "botanical", "urban", "sci-fi", "watercolor", "chiaroscuro", "geometric", "industrial", "pixelated", "celestial", "gothic", "comic", "pop", "steampunk", "collage", "natural", "mythical", "mystical", "textured", "illuminated", "dynamic", "layered", "sculptural", "organic", "fractal", "pastoral", "illustrative", "monochromatic", "fiery", "frozen", "ceramic", "fluid", "dystopian", "ethereal", "mechanical", "atmospheric", "ceremonial", "cultural", "symbolic", "majestic", "serene", "intricate", "dramatic", "playful", "cinematic", "mythological", "cosmic", "rustic", "energetic", "elegant", "fictional", "whirlwind", "mysterious", "eccentric", "sunny", "vivid", "architectural", "enigmatic", "horror"])


    _subject_types = ["random"] + sorted([
        "Adventurers", "Aliens", "Alchemists", "Amazons", "Angels", "Animals", "Archaeologists", "Archers", "Artists", "Astronauts", "Athletes", "Aviators", "Bakers", "Barbarians", "Beekeepers", "Bikers", "Birds", "Black Men", "Black Women", "Blacksmiths", "Boxers", "Builders", "Bystanders", "Captains", "Car Enthusiasts", "Celebrities", "Centaurs", "Chefs", "Children", "Chinese Men", "Chinese Women", "Climbers", "Clowns", "Coders", "Comedians", "Cowboys", "Craftsmen", "Crystals", "Couples", "Criminals", "Cyberpunk Characters", "Cyclists", "Dancers", "Demons", "Detectives", "Dinosaurs", "Divers", "Doctors", "Dreamers", "Druids", "Elves", "Engineers", "Explorers", "Fairies", "Family", "Farmers", "Fantasy Creatures", "Farm Animals", "Fencers", "Firefighters", "Fishermen", "Florists", "Fortune Tellers", "Gardeners", "Geologists", "Ghosts", "Giants", "Gladiators", "Globetrotters", "Guards", "Gymnasts", "Healers", "Herbalists", "Historians", "Hot Men", "Old Women", "Hunters", "Indian Men", "Indian Women", "Inventors", "Japanese Men", "Japanese Women", "Jewelers", "Jockeys", "Knights", "Korean Men", "Korean Women", "Latin Men", "Latin Women", "Librarians", "Lifeguards", "Lumberjacks", "Magicians", "Mariners", "Mechanic Bots", "Mechanics", "Men", "Mermaids", "Merchants", "Messengers", "Middle Eastern Men", "Middle Eastern Women", "Miners", "Monarchs", "Monks", "Mountain Climbers", "Musicians", "Mythical Heroes", "Native American Men", "Native American Women", "Navigators", "Ninjas", "Nurses", "Painters", "Pilots", "Pirates", "Plumbers", "Poets", "Politicians", "Princes", "Princesses", "Professors", "Puppeteers", "Rangers", "Rebels", "Robots", "Royalty", "Samurai", "Scientists", "Sculptors", "Sentinels", "Shepherds", "Shoemakers", "Singers", "Sledders", "Sorcerers", "Spies", "Soldiers", "South Asian Men", "South Asian Women", "Steampunk Characters", "Storytellers", "Street Artists", "Street Performers", "Students", "Swimmers", "Tailors", "Teachers", "Thieves", "Time Travelers", "Trainers", "Travelers", "Trolls", "Vampires", "Veterans", "Villains", "Warriors", "Watchmakers", "Werewolves", "Whalers", "White Men", "White Women", "Wizards", "Women", "Women African", "Women Latin", "Witches", "Wrestlers", "Youths", "Zealots", "Zombies", "Motorcycles", "Trucks", "Aircraft", "Cars", "Fighter Jets", "Space", "Spacecraft", "Guns", "Ships", "Tanks", "Food", "Architecture", "Logos", "Designs", "Rooms", "Wallpaper", "Underwater"
    ])
    _types_of_images = ["random"] + sorted([
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

    def __init__(self):
        self.prompt_generator = HyperComplexImageDescriptionGenerator()

    @classmethod
    def INPUT_TYPES(cls):
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""})
        int_type = ("INT", {"default": random.randint(10**14, 10**15), "min": 10**14, "max": 10**15})
        boolean_type = ("BOOLEAN", {"default": False})
        list_category = (cls._categories, {"default": "subjects"})
        list_mood_type = (cls._mood_types, {"default": "none"})
        list_color_type = (cls._color_types, {"default": "none"})
        list_composition_type = (cls._composition_types, {"default": "none"})
        list_detail_type = (cls._detail_types, {"default": "none"})
        list_weather_type = (cls._weather_types, {"default": "none"})
        list_time_of_day_type = (cls._time_of_day_types, {"default": "none"})
        list_object_type = (cls._object_types, {"default": "none"})
        list_style_type = (cls._style_types, {"default": "none"})
        list_subject_type = (cls._subject_types, {"default": "random"})
        list_type_of_image = (cls._types_of_images, {"default": "random"})
        return {"required": {
            "positive": string_type,
            "short_text": string_type,
            "seed": int_type,
            "expand": boolean_type,
            "subject_only": boolean_type,
            "category": list_category,
            "mood_type": list_mood_type,  # New input for mood type
            "color_type": list_color_type,  # New input for color type
            "composition_type": list_composition_type,  # New input for composition type
            "detail_type": list_detail_type,  # New input for detail type
            "weather_type": list_weather_type,
            "time_of_day_type": list_time_of_day_type,
            "object_type": list_object_type,
            "style_type": list_style_type,
            "subject_type": list_subject_type,
            "type_of_image": list_type_of_image,
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("generated_prompts", "seed")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman/Prompt"

    def select_prompt(self, positive, short_text, seed, expand, subject_only, category, mood_type, color_type, composition_type, detail_type, weather_type, time_of_day_type, object_type, style_type, subject_type, type_of_image):
        # Correctly pass parameters
        generated_prompt = self.prompt_generator.generate_prompt(
            seed=seed,
            subject_type=subject_type,
            category=category,
            replacement=short_text if expand else None,
            image_type=type_of_image,
            mood_type=mood_type if mood_type != "random" else None,
            color_type=color_type if color_type != "random" else None,
            composition_type=composition_type if composition_type != "random" else None,
            detail_type=detail_type if detail_type != "random" else None,
            weather_type=weather_type if weather_type != "random" else None,
            time_of_day_type=time_of_day_type if time_of_day_type != "random" else None,
            object_type=object_type if object_type != "random" else None,
            style_type=style_type if style_type != "random" else None,
            subject_only=subject_only,
            short_text=short_text
        )
        if expand:
            combined_prompts = f"{generated_prompt}\n{positive}"
        else:
            combined_prompts = generated_prompt
        return (combined_prompts, seed)