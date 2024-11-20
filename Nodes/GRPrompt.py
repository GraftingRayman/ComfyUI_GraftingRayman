import torch
import random
from clip import tokenize


class GRPromptSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        clip_type = ("CLIP",)
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True})
        return {"required": {
            "clip": clip_type,
            **{f"positive_a{i}": string_type for i in range(1, 7)},
            "always_a1": string_type,
            "negative_a1": string_type,
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompts")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman/Prompt"

    def select_prompt(self, clip, **kwargs):
        select_prompt = kwargs["select_prompt"]
        positive_clip = kwargs[f"positive_a{select_prompt}"]
        always_a1 = kwargs["always_a1"]
        negative_a1 = kwargs["negative_a1"]

        positive = f"{positive_clip}, {always_a1}"
        prompts = f"positive:\n{positive}\n\nnegative:\n{negative_a1}"

        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)

        return ([[condP, {"pooled_output": pooledP}]], [[condN, {"pooled_output": pooledN}]], prompts)

class GRPromptSelectorMulti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "positive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a2": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "positive_a3": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a4": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a5": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a6": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "alwayspositive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "negative_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            }}


    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive1","positive2","positive3","positive4","positive5","positive6","negative",)
    FUNCTION = "select_promptmulti"
    CATEGORY = "GraftingRayman/Prompt"

        


    def select_promptmulti(self, clip, positive_a1, positive_a2, positive_a3, positive_a4, positive_a5, positive_a6, alwayspositive_a1, negative_a1):

        positive1 = positive_a1 + ", " + alwayspositive_a1
        positive2 = positive_a2 + ", " + alwayspositive_a1
        positive3 = positive_a3 + ", " + alwayspositive_a1
        positive4 = positive_a4 + ", " + alwayspositive_a1
        positive5 = positive_a5 + ", " + alwayspositive_a1
        positive6 = positive_a6 + ", " + alwayspositive_a1
        tokensP1 = clip.tokenize(positive1)
        tokensP2 = clip.tokenize(positive2)
        tokensP3 = clip.tokenize(positive3)
        tokensP4 = clip.tokenize(positive4)
        tokensP5 = clip.tokenize(positive5)
        tokensP6 = clip.tokenize(positive6)
        tokensN1 = clip.tokenize(negative_a1)
        condP1, pooledP1 = clip.encode_from_tokens(tokensP1, return_pooled=True)
        condP2, pooledP2 = clip.encode_from_tokens(tokensP2, return_pooled=True)
        condP3, pooledP3 = clip.encode_from_tokens(tokensP3, return_pooled=True)
        condP4, pooledP4 = clip.encode_from_tokens(tokensP4, return_pooled=True)
        condP5, pooledP5 = clip.encode_from_tokens(tokensP5, return_pooled=True)
        condP6, pooledP6 = clip.encode_from_tokens(tokensP6, return_pooled=True)
        condN1, pooledN1 = clip.encode_from_tokens(tokensN1, return_pooled=True)

        return ([[condP1, {"pooled_output": pooledP1}]],[[condP2, {"pooled_output": pooledP2}]],[[condP3, {"pooled_output": pooledP3}]],[[condP4, {"pooled_output": pooledP4}]],[[condP5, {"pooled_output": pooledP5}]],[[condP6, {"pooled_output": pooledP6}]],[[condN1, {"pooled_output": pooledN1}]],)

class GRPromptHub:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive1": ("CONDITIONING", )
            },
            "optional": {
                "positive2": ("CONDITIONING", ),
                "positive3": ("CONDITIONING", ),
                "positive4": ("CONDITIONING", ),
                "positive5": ("CONDITIONING", ),
                "positive6": ("CONDITIONING", ),
                "negative1": ("CONDITIONING", ),
                "negative2": ("CONDITIONING", ),
                "negative3": ("CONDITIONING", ),
                "negative4": ("CONDITIONING", ),
                "negative5": ("CONDITIONING", ),
                "negative6": ("CONDITIONING", )
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "combine"
    CATEGORY = "GraftingRayman/Prompt"
    
    def combine(self, positive1, positive2=None, positive3=None, positive4=None, positive5=None, positive6=None,
                negative1=None, negative2=None, negative3=None, negative4=None, negative5=None, negative6=None):
        positive_result = positive1
        if positive2 is not None:
            positive_result += positive2
        if positive3 is not None:
            positive_result += positive3
        if positive4 is not None:
            positive_result += positive4
        if positive5 is not None:
            positive_result += positive5
        if positive6 is not None:
            positive_result += positive6
        
        negative_result = negative1 if negative1 is not None else 0
        if negative2 is not None:
            negative_result += negative2
        if negative3 is not None:
            negative_result += negative3
        if negative4 is not None:
            negative_result += negative4
        if negative5 is not None:
            negative_result += negative5
        if negative6 is not None:
            negative_result += negative6
        
        return (positive_result, negative_result)


class GRPrompty:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        clip_type = ("CLIP",)
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True})
        return {"required": {
            "clip": clip_type,
            **{f"positive_a{i}": string_type for i in range(1, 9)},  # Include positive_a1 to positive_a8
            "always_a1": string_type,
            "negative_a1": string_type,
            "select_prompts": ("STRING", {"default": "1", "multiline": False}),
            "randomize": ("BOOLEAN", {"default": False}),
            "multi_prompt": ("BOOLEAN", {"default": True}),  # New input
            "seed": ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompts")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman/Prompt"

    def select_prompt(self, clip, **kwargs):
        select_prompts_str = kwargs["select_prompts"]
        always_a1 = kwargs["always_a1"]
        negative_a1 = kwargs["negative_a1"]
        randomize = kwargs["randomize"]
        multi_prompt = kwargs["multi_prompt"]
        seed = kwargs["seed"]

        # Set the seed for reproducibility if `randomize` is true
        if randomize:
            random.seed(seed)
            select_prompts = random.sample(range(1, 9), k=random.randint(1, 8)) if multi_prompt else [random.choice(range(1, 9))]
        else:
            try:
                # Parse the prompt selection string
                select_prompts = [int(i) for i in select_prompts_str.split(",") if i.strip().isdigit() and 1 <= int(i.strip()) <= 8]
                if not multi_prompt:
                    select_prompts = [random.choice(select_prompts)]  # Select only one prompt if multi_prompt is False
            except ValueError:
                raise ValueError("select_prompts should contain comma-separated numbers between 1 and 8.")

        # Combine the selected positive prompts
        positive_clips = [kwargs[f"positive_a{i}"] for i in select_prompts]
        combined_positive = ", ".join(positive_clips) + f", {always_a1}"
        
        # Formulate prompts output
        prompts = f"positive:\n{combined_positive}\n\nnegative:\n{negative_a1}"

        # Tokenization and encoding
        tokensP = clip.tokenize(combined_positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)

        return ([[condP, {"pooled_output": pooledP}]], [[condN, {"pooled_output": pooledN}]], prompts)

class HyperComplexImageDescriptionGenerator:
    """Generates hyper-complex image descriptions with vast variety."""
    def __init__(self):
        self.subjects = [
            "a tranquil forest clearing bathed in golden morning light",
            "a serene lake surrounded by misty mountains",
            "a vibrant coral reef teeming with marine life",
            "a towering volcano erupting under a twilight sky",
            "a sunflower field stretching endlessly under a golden sun",
            "a peaceful river winding through a lush green valley",
            "a bustling medieval market square filled with merchants and performers",
            "an abandoned factory overtaken by vines and rust",
            "a grand palace adorned with sparkling gemstones and golden domes",
            "a futuristic space station orbiting a distant planet",
            "an ancient Roman amphitheater crumbling under time’s weight",
            "a magical castle floating above a sea of clouds",
            "a mysterious cave with glowing crystalline formations",
            "a sprawling cyberpunk cityscape drenched in neon light",
            "a snowy village illuminated by warm glowing lanterns",
            "a stormy ocean with towering waves crashing against cliffs",
            "a hidden waterfall cascading into a crystal-clear lagoon",
            "a mystical forest with gigantic, luminescent mushrooms",
            "a tranquil Japanese garden with blooming cherry blossoms",
            "a majestic eagle soaring over rugged mountain peaks",
            "an enchanting meadow filled with colorful wildflowers",
            "a sprawling desert under a sky filled with twinkling stars",
            "a dragon circling an ancient stone tower on a foggy hill",
            "a space rover exploring a distant, barren alien world",
            "a cozy cottage surrounded by a field of lavender flowers",
            "a dark enchanted forest with a glowing portal between trees",
            "an underwater city with glass domes and bioluminescent lights",
            "a bustling futuristic market filled with robots and aliens",
            "a majestic white horse running through a sunlit meadow",
            "a pirate ship sailing across a turquoise tropical sea",
            "a lonely lighthouse standing tall against a stormy sky",
            "a peaceful temple nestled high in the snowy mountains",
            "a fantasy realm with floating islands and waterfalls",
            "a sunflower patch being gently swayed by a summer breeze",
            "a hot air balloon festival with dozens of colorful balloons",
            "a wild savannah with lions basking under the setting sun",
            "an ancient library filled with dusty, towering bookshelves",
            "a caravan of camels crossing vast golden sand dunes",
            "a serene fjord surrounded by steep, snow-covered cliffs",
            "a sunset over a city skyline viewed from a quiet rooftop",
            "a quaint village market selling an abundance of fresh fruits",
            "a fairy ring in a foggy woodland during the early dawn",
            "a cozy cabin with smoke rising from the chimney in winter",
            "an old wooden shipwreck washed ashore on a rocky beach",
            "a scenic vineyard with grapevines ready for harvest",
            "a wide river delta with small boats traveling in the mist",
            "a vast field of wheat swaying gently in the wind at dusk",
            "a golden temple gleaming under a bright blue tropical sky",
            "a bustling café in Paris during a rainy afternoon",
            "an abandoned train station with ivy-covered platforms",
            "a glowing aurora lighting up the night sky over a frozen lake",
            "a tranquil zen rock garden surrounded by tall bamboo",
            "a vibrant rainbow arcing over a calm, misty harbor",
            "a secluded monastery perched on a rugged mountain ledge",
            "a bustling harbor filled with colorful fishing boats",
            "a dark cave with ancient drawings illuminated by torchlight",
            "a field of glowing fireflies under a starlit sky",
            "an otherworldly canyon with vibrant, swirling rock formations",
            "a peaceful beach with palm trees swaying in the ocean breeze",
            "a frosty tundra with a lone polar bear crossing the ice",
            "a magical meadow where unicorns graze by a clear brook",
            "a quiet countryside lane lined with blooming wisteria trees",
            "an old windmill standing alone in a vast open field",
            "a bustling airport terminal with planes taking off in the distance",
            "a serene lagoon surrounded by jagged volcanic cliffs",
            "a snowy mountain trail with fresh tracks leading into the distance",
            "a vibrant carnival with twinkling lights and joyful laughter",
            "a surreal dreamscape filled with floating clocks and staircases",
            "a futuristic metropolis with hovering cars and illuminated skyscrapers",
            "a quiet farmstead with a barn glowing warmly under the setting sun",
            "a ship sailing through an ice-filled Arctic expanse",
            "a ruined castle with ivy creeping up its weathered stone walls",
            "a secret garden hidden behind an ornate iron gate",
            "a vast orchard bursting with ripe, colorful fruits",
            "a cozy bookstore with shelves reaching up to the ceiling",
            "a peaceful island surrounded by turquoise waters and coral reefs",
            "a dense jungle with a hidden ancient temple at its heart",
            "a magical library with books floating and glowing with light",
            "a moonlit battlefield where ghostly warriors stand frozen in time",
            "a bustling city street with vibrant street art on every wall",
            "a serene arctic bay under the glow of the midnight sun",
            "a mystical portal in a stone archway leading to another dimension",
            "a wildflower meadow where bees and butterflies flit about",
            "a grand opera house illuminated against a starry night",
            "a lively summer fair with ferris wheels and cotton candy",
            "a dense fog rolling over an eerie abandoned village",
            "a lively rainforest canopy bustling with colorful birds",
            "a train winding through a picturesque mountain landscape",
            "a dramatic canyon with a rushing river cutting through it",
            "a peaceful monastery garden with koi ponds and lanterns",
            "a volcanic island surrounded by steaming geothermal pools",
            "a mysterious swamp with glowing will-o'-the-wisps",
            "a celestial observatory atop a mountain under a vast galaxy",
            "a tranquil lake reflecting a vibrant autumn forest",
            "a rolling countryside dotted with sheep grazing peacefully",
            "a thriving coral atoll teeming with exotic marine life",
            "a stark desert landscape with towering rock formations",
            "a sprawling vineyard under a warm, golden sunset",
            "a castle courtyard with knights training under a bright sky",
            "a dramatic coastal cliffside with waves crashing below",
            "a remote lighthouse beaming out into a stormy night",
            "a quaint fishing village with colorful houses along the shore",
            "a highland glen with a sparkling stream cutting through the valley",
            "an intricate labyrinth filled with hidden treasures and traps",
            "a glimmering ice cave with crystalline walls reflecting every hue",
            "a tropical waterfall surrounded by vibrant orchids",
            "a desert oasis with crystal-clear waters and towering palms",
            "a floating city above the clouds with waterfalls cascading below",
            "a hidden grotto filled with glowing fungi",
            "a shimmering field of snow under the northern lights",
            "a craggy mountain peak piercing through the clouds",
            "a sprawling treehouse connected by intricate rope bridges",
            "a secluded beach with bioluminescent waves lapping at the shore",
            "a dense bamboo forest rustling in a gentle breeze",
            "an ancient temple overrun with flowering vines",
            "a colorful coral lagoon teeming with exotic fish",
            "a tranquil meadow under a sky streaked with rainbows",
            "a cave opening overlooking a vast, colorful valley",
            "a glacial river cutting through a canyon of ice",
            "a rustic wooden dock jutting out into a tranquil lake",
            "a bustling spaceport with starships taking off into a purple sky",
            "a valley dotted with bright red poppies swaying in the wind",
            "a volcanic plain glowing with rivers of molten lava",
            "a winter wonderland of frosted trees and frozen lakes",
            "a quiet chapel atop a rolling hill under a clear blue sky",
            "a mystical forest with trees glowing softly in the dark"
        ]


        self.moods = [
            "serene", "vibrant", "chaotic", "mysterious", "melancholic",
            "dramatic", "dreamlike", "nostalgic", "adventurous", "ominous",
            "uplifting", "tense", "eerie", "romantic", "hopeful",
            "whimsical", "energetic", "contemplative", "majestic", "peaceful",
            "solemn", "foreboding", "cheerful", "tranquil", "empowering",
            "playful", "introspective", "joyful", "haunting", "calming",
            "bright", "sinister", "ethereal", "spiritual", "quirky",
            "poignant", "breathtaking", "mystical", "bold", "reverent",
            "intense", "lighthearted", "euphoric", "brooding", "suspenseful",
            "comforting", "dynamic", "serendipitous", "fierce", "meditative",
            "melodic", "explorative", "celestial", "gritty", "reassuring",
            "unsettling", "wistful", "jubilant", "isolated", "provocative",
            "sacred", "festive", "inspiring", "desolate", "opulent",
            "graceful", "regal", "lively", "humble", "surreal",
            "mellow", "frantic", "pristine", "delicate", "thoughtful",
            "uplifting", "hopeful", "fiery", "harmonious", "ecstatic",
            "mournful", "shadowy", "tranquil", "determined", "enigmatic",
            "jovial", "hushed", "stormy", "radiant", "glorious",
            "ardent", "somber", "elated", "bleak", "boisterous",
            "evocative", "vivid", "ferocious", "lonely", "carefree",
            "relaxed", "sophisticated", "booming", "startling", "idyllic",
            "energetic", "tenacious", "monumental", "spirited", "grateful",
            "elusive", "unwavering", "light", "dark", "colorful",
            "hopeful", "restless", "vulnerable", "cathartic", "anxious",
            "ambitious", "stoic", "charming", "melodramatic", "sorrowful",
            "luminous", "radiant", "fearful", "steadfast", "unpredictable",
            "magnetic", "shimmering", "poised", "heroic", "mischievous",
            "fluid", "warm", "icy", "enigmatic", "provocative",
            "introspective", "impulsive", "conservative", "wild", "visionary",
            "consoling", "oppressive", "delightful", "refreshing", "hypnotic",
            "sublime", "volatile", "courageous", "deliberate", "reluctant",
            "gentle", "harsh", "vulnerable", "playful", "reclusive",
            "exuberant", "dynamic", "steady", "daring", "distant",
            "inviting", "aloof", "pensive", "radiant", "tense",
            "nostalgic", "yearning", "disjointed", "cohesive", "mysterious",
            "lucid", "disoriented", "vivid", "murky", "energetic",
            "melancholy", "spirited", "buoyant", "dark", "ethereal",
            "ominous", "soothing", "vulnerable", "enriching", "unsettling",
            "stimulating", "enigmatic", "exhilarating", "melodious", "somber",
            "uplifting", "transcendent", "serendipitous", "grounded", "abstract",
            "celestial", "phantasmic", "enigmatic", "feral", "whimsical",
            "ethereal", "puzzling", "euphoric", "somnolent", "vibrant",
            "simmering", "adventurous", "rejuvenating", "mellow", "magnanimous",
            "turbulent", "delirious", "cryptic", "tranquil", "emotional",
            "reserved", "fiery", "arduous", "thought-provoking", "minimalist",
            "optimistic", "shattered", "passionate", "frenetic", "timeless",
            "ethereal", "magical", "mystified", "glowing", "indifferent",
            "immersive", "curious", "unexpected", "layered", "detached",
            "structured", "spiritual", "haunted", "jubilant", "symphonic",
            "compassionate", "radiating", "chill", "overwhelming", "gentle",
            "sprawling", "tight", "charming", "mirthful", "wistful",
            "anguished", "cheerful", "exuberant", "reluctant", "gentle",
            "volatile", "stoic", "solemn", "dynamic", "pensive"
        ]

        self.colors = [
            "soft golden hues of sunrise", "deep indigos and purples of twilight",
            "vivid neon blues and greens", "muted sepia tones",
            "fiery oranges and reds", "shimmering silvers and whites",
            "dusky pink and lavender gradients", "dark, shadowy contrasts",
            "bright, saturated primary colors", "subtle, almost monochromatic shades",
            "cool aquamarine and teal tones", "rich emerald greens",
            "warm amber and ochre highlights", "crystal-clear icy blues",
            "the stark black and white of moonlit shadows",
            "pastel tones fading into misty whites",
            "vibrant yellows and sunflower gold",
            "earthy browns and mossy greens",
            "soft peach and coral blends", "glowing turquoise and cerulean hues",
            "bold magentas paired with lime greens", "delicate ivory and champagne shades",
            "stormy grays and slate blues", "vivid crimson and scarlet splashes",
            "burnished bronze and golden undertones", "icy lilac and frosty lavender",
            "rich plum and midnight blue depths", "warm terracotta and sun-baked reds",
            "gentle alabaster and muted beige tones", "lush forest greens and pine tones",
            "subtle rose and blush pinks", "sparkling golds and metallic yellows",
            "cool steel gray with hints of silver", "luminous marigold and saffron shades",
            "sunlit topaz and glinting honey hues", "tranquil sky blue and cloud white",
            "deep charcoal and smoky black", "emerald and jade entwined",
            "burning sunset oranges and pinks", "fresh mint greens with icy accents",
            "shadowy violet and royal purple", "electric cyan with shimmering highlights",
            "subdued olive green and mustard tones", "clear cobalt blue and azure seas",
            "rusty copper and golden brown", "fiery ruby reds with garnet undertones",
            "iridescent pearl whites and opal reflections", "deep oceanic navy and aquamarine",
            "vibrant carnation pink and fuchsia", "silky chocolate browns with caramel swirls",
            "brilliant chartreuse with lemony brightness", "calming periwinkle and sky tones",
            "goldenrod and wheat-field yellows", "midnight black with obsidian sparkles",
            "ethereal lavender and pale lilac whispers", "glowing ember orange and coal black",
            "bright neon pinks and purples", "rich mahogany and walnut shades",
            "bright tangerine and mandarin blends", "soft pastel blues and baby pinks",
            "sun-warmed sand and pale beige tones", "glittering starlight silvers",
            "velvety maroon and crimson tides", "crisp apple greens with lemon zest",
            "frosted teal and icy mint", "sun-drenched yellows and sandy golds",
            "shadowed umber and muted olive", "sparkling sapphire and royal blue depths",
            "calming aquas and soft gray blends", "autumnal oranges and golden browns",
            "tropical coral and watermelon pink", "stormy cobalt and midnight indigo",
            "peach sorbet and apricot gradients", "iridescent bubblegum and pastel hues",
            "opaline shimmer and glossy ivory", "sunset reds blending into twilight purples",
            "sapphire and diamond glints", "electric green and neon yellow blends",
            "muted charcoal and ash tones", "lively lime green and citrusy yellow",
            "coppery gold and fiery bronze", "seafoam green and translucent aquas",
            "dusky apricot and soft cinnamon shades", "arctic silver and frosted steel",
            "flamingo pinks and vivid salmon hues", "bold indigo and amethyst streaks",
            "dreamy lavender and periwinkle pastels", "sun-kissed peach and goldenrod highlights",
            "bright coral reefs of orange and pink", "chilled iceberg whites and blues",
            "deep burgundy and mulberry blends", "glowing amber and flickering gold sparks",
            "ethereal rose gold and soft blush tones", "radiant orchid with fuchsia undertones",
            "misty lavender and silver frost", "stormy teal with flashes of deep indigo",
            "glowing amber sunsets over calm seas", "gold-flecked olive greens and bronze",
            "delicate aquamarine with shimmering silver streaks",
            "soft vanilla cream and hazelnut hues", "vibrant crimson poppies in golden fields",
            "icy periwinkle and sparkling snow white", "sun-dappled emerald and jade greens",
            "shimmering topaz and sapphire ripples", "vivid coral and sunny orange highlights",
            "burnt sienna and toasted cinnamon swirls", "brilliant fuchsia and tropical teal",
            "glimmering turquoise blending into icy blue", "subtle gray-green and eucalyptus tones",
            "pearl white with faint golden sheen", "candy apple red with glossy highlights",
            "metallic platinum and soft gold tones", "deep onyx with hints of obsidian sparkle",
            "blush pinks with coral orange undertones", "sun-washed sandstone and tawny gold",
            "marbled indigo and cobalt with silver streaks",
            "honey-golden highlights with caramel shadows", "sunset purples fading into soft lavender",
            "tangerine sorbet and creamy peach gradients", "luminous lime green and pale chartreuse",
            "stormy navy blues with hints of violet", "shimmering mica and dusty rose tones",
            "mystical amethyst and soft lilac glows", "glowing sunset gold blending into fiery red",
            "ice-blue tones fading into frosty white", "earthy terracotta with muted ochre accents",
            "rich walnut with golden highlights", "deep forest green with emerald sparkles",
            "floral magentas paired with buttercup yellows", "metallic bronze and copper highlights",
            "sunlight glowing through translucent amber tones",
            "bright peacock blues paired with shimmering emerald",
            "shimmering pearl gray with soft moonlit tones", "storm-cloud gray with dashes of deep violet",
            "oceanic teal blending into glowing aquamarine", "rosy peach blending into pale cream",
            "glittering opal hues with shifting pastel tones", "burning copper and fiery orange",
            "velvety black with faint red undertones", "blazing sun-yellow paired with coral orange",
            "cobalt skies transitioning into dusk purple", "brilliant emerald with hints of seafoam",
            "vivid plum and soft mulberry tones", "sparkling silver with pale gold streaks"
        ] 

        self.compositions = [
            "a symmetrical arrangement drawing the eye to the center",
            "a dynamic composition with sweeping lines and strong diagonals",
            "a layered perspective creating a sense of depth and scale",
            "a minimalist layout emphasizing empty space and isolation",
            "a chaotic, overlapping scene full of intricate details",
            "a panoramic view showcasing vast landscapes",
            "a tight close-up highlighting fine textures and patterns",
            "an off-center subject creating tension and movement",
            "a spiral composition that naturally guides the viewer's gaze",
            "an interplay of light and shadow creating dramatic contrast",
            "a balanced layout with harmonious symmetry",
            "a fragmented composition with shards of overlapping elements",
            "an organic flow mimicking natural curves and waves",
            "a radiating pattern centered around a glowing focal point",
            "a triangular composition adding stability and focus",
            "a cascading arrangement of elements creating a sense of motion",
            "a split-screen layout contrasting two opposing themes",
            "a grid-based design emphasizing order and repetition",
            "a radial burst emanating from a central point of energy",
            "a montage blending unrelated images into a cohesive whole",
            "a vignette effect framing the subject with darker edges",
            "a clustered arrangement with tightly packed elements",
            "a loose, scattered arrangement conveying freedom and chaos",
            "a serpentine flow leading the eye in a winding path",
            "a silhouette-based composition using stark contrasts",
            "a high-contrast black-and-white arrangement for impact",
            "a multi-layered collage of transparent and opaque elements",
            "a concentric design with repeated circular patterns",
            "a tilted perspective creating a dynamic and uneasy feeling",
            "a mirrored reflection doubling the subject and surroundings",
            "a horizon line splitting the frame into equal halves",
            "a cross-section revealing layers beneath the surface",
            "a kaleidoscopic arrangement with symmetrical, repeated shapes",
            "a focal point framed by natural elements like branches or arches",
            "a cascading waterfall effect with elements flowing downward",
            "a central glow radiating outward like ripples in water",
            "a fractured glass-like composition with sharp angular divisions",
            "a staggered perspective overlapping multiple planes",
            "a central void surrounded by intricate details",
            "a flowing S-curve leading the viewer's gaze across the frame",
            "a juxtaposition of scale contrasting large and small objects",
            "a rhythmic repetition of similar forms creating a pattern",
            "a minimalist horizon line dividing bold sky and land contrasts",
            "a playful zigzag pattern leading the eye in alternating directions",
            "a strong vertical composition emphasizing height and grandeur",
            "a surreal overlapping of unrelated objects creating a dreamlike effect",
            "a harmonious color-block composition dividing the frame into segments",
            "a diagonal split creating tension between two opposing areas",
            "a blurred motion effect suggesting speed or transition",
            "a meandering river-like flow weaving across the frame",
            "a frame-within-a-frame effect drawing focus inward",
            "a cascading diagonal arrangement suggesting movement and rhythm",
            "a patchwork quilt effect with varied textures and patterns",
            "a strong asymmetrical balance using weighty and light elements",
            "a layered transparency effect revealing hidden shapes beneath",
            "a radial symmetry centered on a natural or artificial focal point",
            "a dramatic sky occupying two-thirds of the frame in a landscape shot",
            "a stair-step arrangement leading the eye upward or downward",
            "a glowing core surrounded by progressively darker gradients",
            "a tessellation effect with repeating geometric forms",
            "a golden-ratio spiral creating a naturally pleasing arrangement",
            "a silhouette-framed opening revealing a dramatic scene beyond",
            "a tension-filled crisscross of lines intersecting at sharp angles",
            "a juxtaposition of organic and mechanical forms in close proximity",
            "a balanced trio of elements forming a visual triangle",
            "a cascading spiral staircase effect drawing the eye downward",
            "a frosted glass-like effect obscuring some elements",
            "a collage-style blend of layered photos or textures",
            "a play of oversized and undersized elements within the frame",
            "a symmetrical mandala-inspired radial design",
            "a hidden subject partially obscured by other elements",
            "a glowing gradient backdrop setting off silhouetted elements",
            "a juxtaposition of sharp focus in one area and soft blur in another",
            "a surrealist floating composition with elements defying gravity",
            "a checkerboard pattern dividing areas of light and shadow",
            "a stepped perspective with layers receding into the distance",
            "a spiral galaxy-like composition with elements swirling outward",
            "a minimalist diagonal bisecting a vibrant color field",
            "a top-down view emphasizing radial or concentric arrangements",
            "a stark negative space contrasting the densely packed focal area",
            "a cascading drapery effect suggesting flowing fabric or waves",
            "a broken horizon with elements crossing into both halves",
            "a whirlwind-like effect with shapes spiraling inward",
            "a playful mosaic of tiny repeated elements forming a larger shape",
            "a bold vertical stripe dividing two contrasting areas",
            "a twisting helix-like composition creating depth and motion"
        ]


        self.details = [
            "gentle ripples in the water reflecting the sky",
            "a soft haze that blurs distant objects",
            "tiny glowing particles floating in the air",
            "intricate carvings etched into stone pillars",
            "a flurry of movement captured mid-action",
            "light filtering through leaves, casting dappled shadows",
            "the texture of weathered wood and rusted metal",
            "the faint glow of distant lanterns in the fog",
            "patterns in the clouds that mirror the terrain below",
            "subtle gradients of color blending seamlessly together",
            "crystals shimmering as light passes through them",
            "footprints left in freshly fallen snow",
            "leaves rustling gently in the breeze",
            "the sparkle of dew on a spiderweb at dawn",
            "a cascade of falling petals caught mid-air",
            "waves crashing rhythmically against the shore",
            "raindrops sliding down a fogged window",
            "the intricate weave of fabric in an ornate tapestry",
            "glinting reflections on a still lake at sunrise",
            "shadows stretching long across the ground at sunset",
            "frost forming delicate patterns on a windowpane",
            "a single droplet of water balancing on a leaf's edge",
            "the glow of embers fading in a dying fire",
            "the shimmer of heat waves rising from hot pavement",
            "bubbles glistening as they float on the breeze",
            "tree roots twisting and curling through the earth",
            "sand ripples shaped by the wind on a dune",
            "the subtle sparkle of mica in a rock face",
            "a spider spinning its web with careful precision",
            "the quiet hum of insects in a meadow at dusk",
            "sunlight refracted into tiny rainbows by a prism",
            "the intricate patterns of veins in a leaf",
            "waves of grass bending in unison with the wind",
            "the smooth surface of polished stone reflecting light",
            "lichen creeping across the surface of ancient bark",
            "stars twinkling faintly against a deep indigo sky",
            "water trickling over rocks in a clear stream",
            "cobwebs swaying gently in an abandoned corner",
            "the glimmer of gold flecks in a shallow creek",
            "mist rising from a forest floor in the early morning",
            "the soft crunch of footsteps on fallen leaves",
            "fireflies flickering like tiny lanterns in the dark",
            "the sharp angles of light through fractured glass",
            "a gust of wind sending leaves swirling upward",
            "snowflakes caught mid-air, each one unique",
            "the glow of moonlight reflected on still water",
            "drops of honey dripping slowly from a comb",
            "dust motes swirling in a sunbeam through a window",
            "puddles rippling as rain continues to fall",
            "patterns etched by time into ancient cliffs",
            "the soft glow of bioluminescent algae in the dark",
            "lichen-stained rocks nestled in a forest clearing",
            "soft pinks and purples blending in a twilight sky",
            "sparkling frost crystals on the tips of grass blades",
            "the gentle sway of tall wildflowers in the breeze",
            "the sheen of oil creating rainbows on wet pavement",
            "tiny streams of water carving paths through sand",
            "the rough texture of unpolished stone walls",
            "the flicker of candlelight casting dancing shadows",
            "a single feather floating gently toward the ground",
            "the soft crackle of ice melting in spring sunlight",
            "sunbeams breaking through stormy gray clouds",
            "the soft rustle of reeds in a lakeside breeze",
            "small fish darting in clear, shallow waters",
            "golden pollen drifting lazily through the air",
            "the play of light and shadow on rippling water",
            "the fine mist of a waterfall lingering in the air",
            "sparkling reflections in scattered morning dew",
            "a flock of birds silhouetted against a fiery sunset",
            "glistening icicles catching the light as they melt",
            "wind carving grooves into the surface of snowdrifts",
            "a cat stretching luxuriously in a patch of sunlight",
            "the steady drip of water from a stalactite",
            "a dragonfly's wings shimmering in the sunlight",
            "the swirling patterns of ink dropped into water",
            "the sharp glint of a blade catching the light",
            "fluffy clouds casting shadows over rolling hills",
            "a curtain billowing gently in the breeze",
            "reflections of city lights shimmering on wet streets",
            "soft ripples spreading outward from a skipping stone",
            "a butterfly delicately perched on a flower",
            "the smooth texture of river rocks worn by water",
            "vibrant green moss blanketing a forest floor",
            "smoke curling lazily upward from a chimney",
            "a flock of starlings moving in synchronized waves",
            "the way snow glistens under the cold winter sun",
            "a whisper of mist clinging to a mountain peak",
            "sunlight filtering through a canopy of blossoms",
            "the flicker of lightning illuminating a darkened sky",
            "a glassy tide pool brimming with marine life",
            "a stream of sparks flying from a blacksmith's hammer",
            "the subtle shift of shadows as clouds drift by",
            "the soft murmur of waves lapping against a boat"
        ]

        self.weather = [
            "a gentle rain falling under gray skies",
            "a fierce thunderstorm with flashes of lightning",
            "a calm, cloudless day with bright sunlight",
            "a snowy blizzard obscuring the view",
            "a misty morning with dew glistening on every surface",
            "a windy evening with leaves swirling through the air",
            "a humid, tropical heat that saturates the scene",
            "a soft drizzle that creates shimmering reflections",
            "a golden sunset with streaks of orange and pink",
            "an eerie fog that cloaks the landscape in mystery",
            "a clear, star-filled night with a cool breeze",
            "a heavy hailstorm rattling rooftops",
            "a calm after a storm with rainbows arcing overhead",
            "a scorching afternoon with heatwaves rippling in the air",
            "a brisk autumn day with leaves tumbling to the ground",
            "a chilly dawn with frost coating every surface",
            "a turbulent hurricane with howling winds and driving rain",
            "a tranquil snowfall with large, fluffy flakes drifting down",
            "a hazy summer evening with the sun glowing low on the horizon",
            "a sudden downpour soaking everything in moments",
            "a dust storm swirling through a barren desert landscape",
            "a crisp winter morning with sunlight glinting off ice",
            "a tropical monsoon drenching the vibrant landscape",
            "a brilliant sunrise painting the sky in vivid colors",
            "a high-altitude breeze brushing against rugged peaks",
            "a cold, overcast afternoon with a sense of impending rain",
            "a peaceful spring rain nurturing budding flowers",
            "a brooding sky filled with dark, rolling storm clouds",
            "a calm ocean reflecting a pastel-colored dawn",
            "a gale-force wind whipping waves into whitecaps",
            "a humid evening with cicadas buzzing in the background",
            "a summer thunderstorm lighting up the night sky",
            "a heavy snowfall blanketing the earth in silence",
            "a mild drizzle accompanied by the smell of wet earth",
            "a dazzling lightning storm over a vast open plain",
            "a muggy afternoon with clouds building in the distance",
            "a brisk coastal wind carrying the scent of saltwater",
            "a serene twilight where the sky shifts from pink to deep blue",
            "a sweltering desert heat under a blazing sun",
            "a foggy forest trail with visibility fading into the unknown",
            "a vibrant aurora dancing across a polar sky",
            "a warm spring breeze carrying the scent of blooming flowers",
            "a relentless downpour flooding cobblestone streets",
            "a hazy day where the sun struggles to break through",
            "a dramatic storm front rolling in over open fields",
            "a shimmering mirage in the heat of the midday sun",
            "a bone-chilling arctic wind cutting through the air",
            "a mild summer evening with fireflies blinking in the dark",
            "a dense fog rolling off a lake at dawn",
            "a sunny day with cotton-candy clouds dotting the sky",
            "a heavy mist rising from a waterfall in the morning light",
            "a relentless sandstorm engulfing a desert caravan",
            "a powerful thunderclap shaking the ground underfoot",
            "a gentle snowfall transforming the landscape into a winter wonderland",
            "a freezing rain coating everything in a layer of ice",
            "a sudden gust of wind scattering papers in the city streets",
            "a muggy swamp air thick with moisture and buzzing insects",
            "a tranquil evening where the horizon glows with fading light",
            "a rainstorm battering against the windows of a quiet house",
            "a clear mountain morning with a chill in the air",
            "a thunderstorm at sea with waves crashing against the hull",
            "a glowing horizon promising a scorching day ahead",
            "a bitterly cold night under a full moon",
            "a sudden flurry of snow enveloping the trees in white",
            "a calm lake mirroring the pale blue sky above",
            "a relentless storm pounding against a forest canopy",
            "a faint mist lingering over a dew-soaked meadow",
            "a bright and breezy day with occasional gusts",
            "a humid jungle day with rain dripping from every leaf",
            "a roaring windstorm swirling debris across the landscape",
            "a tranquil, windless dusk as the stars begin to appear",
            "a steamy day in the tropics with heavy air and distant thunder",
            "a heavy fog creeping in as evening falls",
            "a golden-hour glow casting long shadows on the ground",
            "a turbulent snowstorm obscuring a distant cabin",
            "a chilly afternoon where clouds threaten an incoming storm",
            "a dazzling display of lightning illuminating the night sky",
            "a dry, hot wind blowing across the savannah",
            "a peaceful night with the faint sound of waves lapping the shore",
            "a frosty evening where breath hangs in the air",
            "a rainy day with the comforting sound of droplets on the roof",
            "a bright day where the sunlight glints off every surface",
            "a howling wind rushing through a mountain pass",
            "a cool, refreshing breeze rustling through tall grass",
            "a cold drizzle mixing with the scent of pine in the forest",
            "a morning mist lifting to reveal a sunlit valley",
            "a dazzling rainbow stretching across the horizon after the rain",
            "a rolling thunder echoing across a sprawling plain",
            "a silent snowfall muffling all sound around",
            "a humid city evening where the pavement radiates heat",
            "a vivid sunset where fiery reds fade into soft purples",
            "a serene winter day with sunlight glinting off icicles"
        ]

        self.time_of_day = [
            "early morning as the first rays of sunlight break through",
            "midday with the sun directly overhead",
            "late afternoon with long, golden shadows",
            "dusk as the sky transitions to deep blues and purples",
            "night under a star-filled sky",
            "a moonlit evening with soft silvery light",
            "twilight with the last glow of the sun on the horizon",
            "a stormy dusk with roiling clouds overhead",
            "a vibrant sunrise painting the sky in fiery colors",
            "a tranquil midnight with only the sound of nature",
            "the pre-dawn hour when the world is bathed in faint gray light",
            "a foggy morning with the sun barely visible through the mist",
            "a cloudy afternoon casting diffuse light over the landscape",
            "sunset as the horizon glows with fiery reds and oranges",
            "a brisk evening as the air cools and the sky deepens in color",
            "a starlit night where constellations shine brightly overhead",
            "a golden morning with sunlight streaming through the trees",
            "a hot afternoon with the sun blazing in a cloudless sky",
            "a crisp autumn dawn with frost sparkling on the ground",
            "a hazy midday with the sun partially obscured by thin clouds",
            "a vibrant sunset reflecting in the calm waters of a lake",
            "a rainy morning with droplets clinging to every surface",
            "an overcast afternoon with muted, shadowless light",
            "a twilight sky streaked with pinks and lavender hues",
            "a chilly dawn as the first light of day creeps over the horizon",
            "an evening filled with the glow of fireflies in the fading light",
            "a dark and stormy night with flashes of lightning illuminating the sky",
            "a moonlit forest casting intricate shadows on the ground",
            "a serene midnight where the world seems to stand still",
            "a sunny morning with dew glistening on blades of grass",
            "a dusky evening as bats flit through the twilight sky",
            "a cold afternoon where the sun hangs low in the sky",
            "a glowing dawn with the sky transitioning from indigo to gold",
            "a windy evening with clouds racing across the darkening sky",
            "a humid afternoon where the light feels thick and heavy",
            "a tropical sunset with brilliant shades of orange and magenta",
            "a peaceful midnight under a full moon casting silver light",
            "a rainy afternoon where the sky seems perpetually gray",
            "a starless night with heavy clouds obscuring the heavens",
            "a soft, pastel dawn as the horizon brightens ever so slowly",
            "an electric sunset following a thunderstorm, vivid and fleeting",
            "a cold winter morning with the sun rising over frosted hills",
            "a sweltering afternoon with the sun dominating the landscape",
            "a mystical twilight with the horizon aglow in soft purple hues",
            "an eerie dusk as fog rolls in and engulfs the scene",
            "a radiant dawn with beams of sunlight cutting through the mist",
            "a breezy afternoon with leaves rustling under a mild sun",
            "a peaceful evening as the first stars begin to twinkle",
            "a golden-hour sunset painting the world in warm, soft tones",
            "a windy twilight with the air cool and the sky vibrant",
            "a brilliant sunrise breaking through stormy clouds",
            "a tranquil night with a slight breeze rustling the trees",
            "a deep midnight where the world is cloaked in darkness",
            "a glowing evening where city lights mingle with the fading sun",
            "an icy dawn as the cold blue of morning overtakes the stars",
            "a radiant afternoon as the sun casts sharp shadows across the ground",
            "a serene sunset where the ocean reflects every color in the sky",
            "a humid evening with the last light of the day lingering in the air",
            "a foggy dusk as the world takes on a muted, dreamlike quality",
            "a clear morning with the sun climbing steadily in the sky",
            "a summer night with the scent of blooming flowers in the warm air",
            "a chilly twilight as the day transitions to night with a crisp chill",
            "a hazy sunset where the sun’s glow diffuses across the horizon",
            "an enchanting evening under a crescent moon’s soft glow",
            "a winter dawn as the sun struggles to warm the frosty landscape",
            "a rainy night with the sound of droplets echoing in the silence",
            "a dramatic sunset as storm clouds give way to brilliant colors",
            "a cloudless midnight with the Milky Way clearly visible",
            "a golden morning as birds begin to sing and the world awakens",
            "a shadowy dusk as the sky turns charcoal with hints of orange",
            "a crisp spring morning with sunlight cutting through cool air",
            "a dark evening where city lights sparkle in the distance",
            "a snowy twilight with the sky blending into the white-covered ground",
            "an overcast sunrise where soft light slowly brightens the horizon",
            "a vivid dawn with streaks of red and yellow heralding the day",
            "a vibrant afternoon with the sun at its zenith and shadows sharp",
            "a tranquil evening as a gentle breeze rustles the water’s surface",
            "a mystical midnight with auroras dancing across the sky",
            "a bright morning with every color enhanced by the clear light",
            "a gentle dusk with the day’s warmth lingering in the air",
            "a picturesque evening where the clouds glow like molten gold",
            "a humid morning where the air feels thick with anticipation",
            "a brilliant noon sky with white, fluffy clouds drifting lazily",
            "a warm twilight as the first stars appear against a glowing sky",
            "an ethereal sunrise where fog mixes with golden light"
        ]

        self.objects = [
            "a weathered wooden signpost pointing in multiple directions",
            "a lone tree standing against the horizon",
            "a small boat anchored near the shore",
            "an abandoned building overtaken by nature",
            "a colorful kite fluttering in the breeze",
            "a group of people gathered around a crackling fire",
            "a statue of an ancient hero partially buried in the ground",
            "a cluster of wildflowers growing amidst the grass",
            "a flock of birds soaring across the sky",
            "a stone bridge arching over a gently flowing stream",
            "a glowing lantern swinging in the wind",
            "a towering windmill with creaking blades",
            "a shimmering waterfall cascading into a crystal pool",
            "an intricate clocktower with visible gears",
            "a hot air balloon drifting lazily above the landscape",
            "a row of fishing nets drying under the sun",
            "a tattered flag fluttering atop a castle turret",
            "a lone bench sitting under a blooming cherry tree",
            "a rustic barn surrounded by golden fields of wheat",
            "a wooden swing hanging from an ancient oak tree",
            "a stack of firewood neatly arranged by a cabin wall",
            "a cluster of mushrooms growing in the shade of a tree",
            "a rusty bicycle leaning against a stone wall",
            "a cobblestone path winding through a dense forest",
            "a weathered anchor resting on the sandy shore",
            "a brightly painted market stall brimming with produce",
            "a cluster of candles burning softly in the dark",
            "a tangled fishing net caught on the rocks",
            "a birdhouse perched high on a pole swaying in the wind",
            "a wrought-iron gate creaking open to a mysterious garden",
            "a dilapidated fence dividing a wildflower meadow",
            "a telescope pointed towards the stars on a clear night",
            "a crumbling tower silhouetted against a stormy sky",
            "a pair of boots left at the edge of a muddy trail",
            "a cluster of seashells scattered along the shoreline",
            "a hammock tied between two palm trees swaying gently",
            "a solitary lighthouse beaming across a rocky coast",
            "a pile of fallen leaves swirling in an autumn breeze",
            "a hand-painted sign advertising a local festival",
            "a scarecrow standing guard in a sprawling cornfield",
            "a glass bottle half-buried in the sand, its contents unreadable",
            "a snowman with a tilted top hat in a frosty field",
            "a picnic blanket spread under a shady tree",
            "a ship's wheel mounted on a weathered dock",
            "a spiral staircase winding up a stone tower",
            "a stack of hay bales arranged in a barnyard",
            "a wooden crate filled with freshly picked apples",
            "a canopy of lanterns glowing over a cobbled plaza",
            "a faded map pinned to the wall of an explorer's cabin",
            "a bell tower ringing out across a sleepy village",
            "a canoe tied to a rickety dock on a misty lake",
            "a sundial casting long shadows in the late afternoon",
            "a stack of stones balanced precariously on the riverbank",
            "a kite caught in the branches of a tall tree",
            "a rope bridge swaying gently over a deep gorge",
            "a farmer’s market stand brimming with ripe tomatoes",
            "a carousel spinning slowly under twinkling lights",
            "a weathervane shaped like a rooster spinning in the breeze",
            "a glass greenhouse filled with vibrant flowers",
            "a campfire with sparks rising into the starry sky",
            "a market cart loaded with jars of honey and preserves",
            "a bench overlooking a calm lake surrounded by mountains",
            "a lantern-lit street in a quaint European village",
            "a rowboat drifting in the middle of a serene pond",
            "a wind-chime tinkling softly in the breeze",
            "a spiral of driftwood arranged artfully on the beach",
            "a bird's nest nestled in the crook of a tree branch",
            "a market square buzzing with lively activity",
            "a rope swing swaying over a glassy river",
            "a wooden pier stretching out into a foggy harbor",
            "a fishing hut perched precariously on stilts",
            "a sprawling treehouse hidden among thick branches",
            "a glowing camp lantern sitting on a mossy rock",
            "a shepherd’s crook leaning against a stone wall",
            "a rustic bridge crossing over a bubbling brook",
            "a meadow with scattered patches of blooming wildflowers",
            "a clay jug half-buried in the sandy dunes",
            "a field of sunflowers swaying under a bright blue sky",
            "a pair of spectacles resting on an open book",
            "a row of wind turbines spinning slowly in the distance",
            "a fountain in the middle of a bustling city plaza",
            "a clothesline hung with colorful garments fluttering in the wind",
            "a rustic wooden ladder leaning against a fruit tree",
            "a vineyard with rows of grapes basking in golden sunlight",
            "a chest of treasures washed ashore after a storm",
            "a seagull perched atop a weathered wooden post",
            "a small campfire surrounded by logs for seating",
            "a weathered canoe resting on a riverbank",
            "a deserted market square under a pale moonlight",
            "a stone staircase winding through a mossy cliffside",
            "a forest clearing with a bubbling spring in the center",
            "a scarecrow in a pumpkin patch surrounded by crows",
            "a small windmill turning slowly in a gentle breeze",
            "a lantern glowing faintly on a cobblestone street"
        ]

        self.styles = [
            "a hyper-realistic depiction with intricate details",
            "an impressionistic view with soft, blurred edges",
            "a surreal representation with distorted proportions",
            "a minimalist interpretation using bold shapes and colors",
            "a painterly approach with visible brushstrokes",
            "a photorealistic style that mimics a high-resolution photograph",
            "a dreamlike aesthetic with ethereal lighting",
            "a noir-inspired composition with stark contrasts",
            "a whimsical, cartoon-like design",
            "a vintage sepia-toned effect",
            "a futuristic, neon-infused cyberpunk aesthetic",
            "a steampunk-inspired scene with brass and gears",
            "a hand-drawn sketch with pencil and ink outlines",
            "a watercolor painting with delicate washes of color",
            "a dramatic chiaroscuro effect emphasizing light and shadow",
            "an abstract expressionist piece with chaotic, bold strokes",
            "a mosaic-inspired composition with tiny, colorful tiles",
            "a retro 80s-style design with geometric patterns and neon colors",
            "a Gothic aesthetic with dark tones and ornate details",
            "a pop art style with vibrant colors and bold outlines",
            "a cubist interpretation with fragmented and angular shapes",
            "a comic book-inspired scene with halftone shading and speech bubbles",
            "a baroque style rich with elaborate ornamentation",
            "a glitch art aesthetic with digital distortion and artifacts",
            "a vaporwave-inspired design with pastel gradients and retro motifs",
            "a stained glass effect with bold outlines and luminous colors",
            "a rustic, weathered texture evoking old wood and metal",
            "a fantasy style with glowing magical elements",
            "a neon-lit, night cityscape with reflective surfaces",
            "a pixel art style reminiscent of early video games",
            "an Art Deco design with geometric shapes and metallic finishes",
            "a sci-fi aesthetic featuring sleek, metallic surfaces and holograms",
            "a hand-painted mural style with layered, vibrant textures",
            "a sepia-toned photograph with a nostalgic feel",
            "a monochromatic palette emphasizing form and texture",
            "a lush, botanical style with intricate plant details",
            "a psychedelic aesthetic with swirling, bright colors",
            "a cosmic style with stars, galaxies, and glowing nebulae",
            "a romanticized oil painting with rich colors and soft edges",
            "a folk art style with simple, vibrant patterns",
            "a shadow puppet design with silhouette cutouts",
            "an industrial style featuring gritty, metallic textures",
            "a collage effect blending photographs and hand-drawn elements",
            "a neon graffiti-inspired aesthetic with bold, vibrant colors",
            "an anime-inspired design with clean lines and exaggerated expressions",
            "a photographic double exposure effect blending two scenes",
            "a painterly style evoking classic Renaissance art",
            "a digital, vectorized look with sharp, clean lines",
            "a charcoal sketch with rough, smudged textures",
            "a crayon-drawn effect evoking a childlike simplicity",
            "a zentangle-inspired design with intricate repeating patterns",
            "a dreamy, bokeh-filled atmosphere with soft glowing lights",
            "a medieval illuminated manuscript style with intricate borders",
            "a geometric abstraction with bold, interlocking shapes",
            "a gritty, urban aesthetic with street art influences",
            "a fabric-textured design resembling woven threads",
            "a 3D-rendered style with depth and realism",
            "a blueprint effect with white lines on a blue background",
            "a metallic, futuristic design with chrome and reflections",
            "a tribal-inspired design with bold, earthy patterns",
            "a botanical illustration with scientific precision",
            "a cinematic look with widescreen framing and dramatic lighting",
            "a gothic horror aesthetic with eerie, decaying elements",
            "a cel-shaded style resembling animated film graphics",
            "a chiaroscuro technique with intense contrast and drama",
            "a kinetic design filled with motion blur and action",
            "a mythical illustration with fantastical creatures and landscapes",
            "a retro-futuristic style blending vintage and modern elements",
            "a whimsical, fairytale-like style with soft edges and magic",
            "an underwater, shimmering effect with glowing ripples",
            "a dreamy pastel palette creating a soft, calming scene",
            "a dynamic comic panel layout with action-packed scenes",
            "a rough, chalk-drawn effect with bold, textured strokes",
            "a noir style with dramatic shadows and dim light",
            "an ornate Victorian aesthetic with lace and flourishes",
            "a pixelated glitch effect mimicking retro digital errors",
            "a gradient-heavy style emphasizing smooth color transitions",
            "a frozen, icy aesthetic with sharp edges and frosted textures",
            "a galactic theme with stars, black holes, and swirling lights",
            "a steely, industrial look with bolts, gears, and grime",
            "an illuminated holographic design with translucent elements",
            "a medieval tapestry style with woven textures and muted tones",
            "a tropical aesthetic with vibrant greens and glowing sunsets",
            "a desert-inspired palette of ochres, reds, and golden sands",
            "a magical realism effect blending fantasy and reality seamlessly",
            "an underwater coral reef scene with vibrant marine colors",
            "a bold, minimalist poster design with striking typography",
            "a topographical map aesthetic with contour lines and gradients",
            "a rustic watercolor with muted tones and rough edges",
            "an origami-inspired design with sharp folds and geometric patterns"
        ]

        self.types_of_images = sorted([
            "random",  # Random selection of image type
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
            "Vaporwave", "Vintage", "Watercolor", "Woodcut"
        ], key=str.casefold)

    def generate_prompt(self, seed, category=None, replacement=None, image_type=None):
        """
        Generate a detailed prompt with emphasis on the selected type of image.
        Replacement text can be provided for a specific category.
        The seed ensures replicable randomness.
        """
        # Set the random seed for reproducibility
        random.seed(seed)

        # Replace the specified category or select randomly
        subject = replacement if category == "subjects" else random.choice(self.subjects)
        mood = replacement if category == "moods" else random.choice(self.moods)
        color = replacement if category == "colors" else random.choice(self.colors)
        composition = replacement if category == "compositions" else random.choice(self.compositions)
        detail = replacement if category == "details" else random.choice(self.details)
        weather = replacement if category == "weather" else random.choice(self.weather)
        time_of_day = replacement if category == "time_of_day" else random.choice(self.time_of_day)
        obj = replacement if category == "objects" else random.choice(self.objects)
        style = replacement if category == "styles" else random.choice(self.styles)

        # Handle "random" selection for type_of_image
        if image_type == "random" or not image_type:
            image_type = random.choice(self.types_of_images[1:])  # Exclude "random" itself

        # Generate the prompt with the type of image emphasized
        return (
            f"This is a {image_type} image. It captures {subject}. The mood is {mood}, complemented by {color}. "
            f"The composition features {composition}, enhanced by {detail}. "
            f"The weather is described as {weather}, and the time of day is {time_of_day}. "
            f"In the foreground, {obj} stands out, drawing the eye. "
            f"The image is rendered in {style}."
        )


class GRPromptGen:
    """Integration with input system for generating prompts."""
    
    _categories = [
        "subjects", "moods", "colors", "compositions", "details",
        "weather", "time_of_day", "objects", "styles"
    ]
    _types_of_images = sorted([
        "random", "3D Render", "Anime-style", "Black and White", "Cartoon", 
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
        "Vaporwave", "Vintage", "Watercolor", "Woodcut"
    ], key=str.casefold)

    def __init__(self):
        self.prompt_generator = HyperComplexImageDescriptionGenerator()

    @classmethod
    def INPUT_TYPES(cls):
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""})  # Default is empty
        int_type = ("INT", {"default": random.randint(10**14, 10**15 - 1), "min": 10**14, "max": 10**15 - 1})  # 14-digit random seed
        boolean_type = ("BOOLEAN", {"default": False})  # Boolean input for expansion
        list_category = (cls._categories, {"default": "subjects"})
        list_type_of_image = (cls._types_of_images, {"default": "random"})
        return {"required": {
            "positive": string_type,
            "short_text": string_type,
            "seed": int_type,
            "expand": boolean_type,
            "category": list_category,
            "type_of_image": list_type_of_image,
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("generated_prompts", "seed")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman/Prompt"

    def select_prompt(self, positive, short_text, seed, expand, category, type_of_image):
        """Generates the prompt based on the provided inputs."""
        if expand:
            # Generate prompt with a specific category and replacement text
            expanded_text = self.prompt_generator.generate_prompt(
                seed=seed,
                category=category, 
                replacement=short_text, 
                image_type=type_of_image
            )
            combined_prompts = f"{expanded_text}\n{positive}"
        else:
            # Generate a standard prompt
            generated_prompt = self.prompt_generator.generate_prompt(
                seed=seed,
                image_type=type_of_image
            )
            combined_prompts = f"{generated_prompt}\n{positive}"

        return (combined_prompts, seed)