import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import comfy.model_management as mm
import hashlib

class Florence2PromptGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "", "multiline": True}),  # Optional text input for guidance
                "task": (
                    [
                        'caption',
                        'detailed_caption',
                        'more_detailed_caption',
                        'region_caption',
                        'dense_region_caption',
                        'region_proposal',
                        'caption_to_phrase_grounding',
                        'referring_expression_segmentation',
                        'ocr',
                        'ocr_with_region',
                        'docvqa',
                        'prompt_gen_tags',
                        'prompt_gen_mixed_caption',
                        'prompt_gen_analyze',
                        'prompt_gen_mixed_caption_plus',
                    ],
                    {"default": "caption"},
                ),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "model": (
                    [
                        'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                        'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                        'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                        'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
                        'microsoft/Florence-2-base',
                        'microsoft/Florence-2-base-ft',
                        'microsoft/Florence-2-large',
                        'microsoft/Florence-2-large-ft',
                        'HuggingFaceM4/Florence-2-DocVQA',
                        'thwri/CogFlorence-2.1-Large',
                        'thwri/CogFlorence-2.2-Large',
                        'gokaygokay/Florence-2-SD3-Captioner',
                        'gokaygokay/Florence-2-Flux-Large',
                        'NikshepShetty/Florence-2-pixelprose',
                    ],
                    {"default": "MiaoshouAI/Florence-2-large-PromptGen-v2.0"},
                ),
                "image_types": (
                    [
                        "abstract", "adventure", "anime", "architectural", "astrophotography",
                        "black and white", "cartoon", "comic", "concept art", "cultural",
                        "cyberpunk", "documentary", "expressionist", "fairy tale", "fantasy",
                        "fantasy art", "food photography", "futuristic", "gothic", "historical",
                        "horror", "hyperrealistic", "illustration", "impressionist", "industrial",
                        "jungle", "landscape", "macro photography", "military", "minimalist",
                        "modern", "mystery", "mythological", "nature", "night photography",
                        "ocean", "painting", "photorealistic", "pop art", "portrait",
                        "post-apocalyptic", "realistic", "religious", "retro", "romantic",
                        "rural", "sci-fi", "space", "steampunk", "street photography",
                        "sunset", "surreal", "travel", "underwater", "urban", "vintage",
                        "war", "weather", "wildlife",
                    ],
                    {"default": "landscape"},
                ),
                "custom_image_types": ("STRING", {"default": "", "multiline": True}),  # Custom image type description
                "media_type": (["image", "image upscale", "video", "subtle video", "subtle video 2", "sto1o"], {"default": "image"}),  # New input for selecting media type
            },
            "optional": {
                "image": ("IMAGE",),  # Optional image input
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")  # Three outputs: prompt, image type description, and media description
    RETURN_NAMES = ("prompt", "image_type_description", "media_description","ollama")
    FUNCTION = "generate_prompt"
    CATEGORY = "GraftingRayman/Image Processing"

    def hash_seed(self, seed):
        """Hash the seed for reproducibility."""
        seed_bytes = str(seed).encode('utf-8')
        hash_object = hashlib.sha256(seed_bytes)
        return int(hash_object.hexdigest(), 16) % (2**32)

    def load_model(self, model_path, device):
        """Load the processor and model."""
        if os.path.exists(model_path):  # Check if the path is a local directory
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        else:  # Load from Hugging Face Model Hub
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        return processor, model

    def generate_prompt(self, text_input, task, max_new_tokens, num_beams, do_sample, seed, model, image_types, custom_image_types, media_type, image=None):
        """Generate a prompt, image type description, and media description."""
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Load the processor and model
        processor, model = self.load_model(model, device)

        # Set seed for reproducibility
        if seed:
            torch.manual_seed(self.hash_seed(seed))

        # Task prompts
        prompts = {
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        task_prompt = prompts.get(task, '<CAPTION>')

        # Handle image input (if provided)
        if image is not None:
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # Batch dimension is present
                    image = image[0]  # Take the first image in the batch
                image = image.mul(255).byte().cpu().numpy()  # Convert to numpy array and scale to [0, 255]
                image = Image.fromarray(image)  # Convert to PIL Image
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
        else:
            inputs = processor(text=task_prompt, return_tensors="pt").to(device)

        # Generate the prompt
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"] if image is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
            )

        # Decode the generated prompt
        generated_prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_prompt = f"{text_input} {generated_prompt}" if text_input else generated_prompt

        # Image type descriptions
        image_type_descriptions = {
            "abstract": "A vibrant explosion of colors and shapes, blending together in a mesmerizing dance.",
            "adventure": "A thrilling journey through uncharted lands, filled with danger and discovery.",
            "anime": "A colorful and stylized world inspired by Japanese animation and storytelling.",
            "architectural": "A stunning display of architectural design, showcasing the beauty of structures and spaces.",
            "astrophotography": "A breathtaking view of the night sky, capturing the beauty of stars, planets, and galaxies.",
            "black and white": "A timeless and classic monochrome image, emphasizing contrast and texture.",
            "cartoon": "A playful and exaggerated depiction of characters and environments, full of humor.",
            "comic": "A dynamic and action-packed scene, reminiscent of classic comic book art.",
            "concept art": "A creative and imaginative visualization of ideas, often used in film and game design.",
            "cultural": "A rich and diverse representation of cultural heritage and traditions.",
            "cyberpunk": "A neon-lit dystopian future, where technology and humanity collide in chaos.",
            "documentary": "A realistic and informative portrayal of real-life events and subjects.",
            "expressionist": "A bold and emotional portrayal of the world, with exaggerated forms and colors.",
            "fairy tale": "A magical and enchanting world, filled with mythical creatures and wonder.",
            "fantasy": "A realm of magic and wonder, where the impossible becomes reality.",
            "fantasy art": "A vivid and imaginative depiction of fantastical worlds and creatures.",
            "food photography": "A mouth-watering and visually appealing presentation of culinary delights.",
            "futuristic": "A sleek and advanced vision of the future, with cutting-edge technology.",
            "gothic": "A dark and mysterious aesthetic, often associated with medieval and horror themes.",
            "historical": "A glimpse into the past, capturing the essence of a bygone era.",
            "horror": "A dark and terrifying scene, filled with fear and suspense.",
            "hyperrealistic": "An incredibly detailed and lifelike representation, almost indistinguishable from reality.",
            "illustration": "A hand-drawn or digitally created image, often used in books and media.",
            "impressionist": "A soft and dreamy depiction of light and color, capturing fleeting moments.",
            "industrial": "A gritty and mechanical environment, dominated by factories and machinery.",
            "jungle": "A dense and lush tropical forest, teeming with life and vibrant colors.",
            "landscape": "A breathtaking view of nature, with rolling hills and serene skies.",
            "macro photography": "An extreme close-up view, revealing intricate details of small subjects.",
            "military": "A powerful and dramatic depiction of military life and conflict.",
            "minimalist": "A clean and simple composition, focusing on essential shapes and colors.",
            "modern": "A contemporary and sleek design, reflecting current trends and styles.",
            "mystery": "A shadowy and enigmatic scene, filled with intrigue and suspense.",
            "mythological": "A world of gods and legends, where ancient myths come to life.",
            "nature": "A peaceful and untouched wilderness, teeming with life and beauty.",
            "night photography": "A captivating view of the world under the cover of darkness, illuminated by artificial and natural light.",
            "ocean": "A vast and mysterious underwater world, full of marine life and wonder.",
            "painting": "A traditional or digital painting, showcasing artistic skill and creativity.",
            "photorealistic": "An image so detailed and accurate that it resembles a high-resolution photograph.",
            "pop art": "A bold and colorful style inspired by popular culture and mass .",
            "portrait": "A detailed and intimate depiction of a person, capturing their essence.",
            "post-apocalyptic": "A desolate and ruined world, where survival is the only goal.",
            "realistic": "A lifelike and accurate representation of the world, with fine details.",
            "religious": "A spiritual and sacred depiction, often associated with religious themes and iconography.",
            "retro": "A nostalgic throwback to the styles and trends of past decades.",
            "romantic": "A tender and emotional scene, filled with love and passion.",
            "rural": "A peaceful and idyllic countryside, with rolling fields and quiet charm.",
            "sci-fi": "A futuristic and imaginative world, with advanced technology and alien life.",
            "space": "A vast and infinite cosmos, filled with stars, planets, and galaxies.",
            "steampunk": "A retro-futuristic world, where steam-powered technology reigns supreme.",
            "street photography": "A candid and authentic capture of everyday life in urban environments.",
            "sunset": "A warm and serene view of the sun setting on the horizon, casting a golden glow.",
            "surreal": "A dreamlike and fantastical scene, where reality is twisted and strange.",
            "travel": "A vibrant and diverse representation of different cultures and destinations around the world.",
            "underwater": "A serene and mysterious underwater world, full of marine life and beauty.",
            "urban": "A bustling and vibrant cityscape, filled with life and energy.",
            "vintage": "A nostalgic and timeless aesthetic, inspired by the styles of the past.",
            "war": "A dramatic and intense depiction of conflict and its impact on humanity.",
            "weather": "A powerful and dynamic representation of natural weather phenomena.",
            "wildlife": "A vibrant and untamed natural world, teeming with animals and plants.",
        }

        # Get the description for the selected image type
        image_type_description = custom_image_types.strip() or image_type_descriptions.get(image_types, "A creative and unique image.")

        # Media description based on the selected media type
        if media_type == "video":
            media_description = "You are transforming user inputs into descriptive prompts for generating AI Videos. Follow these steps to produce the final description:\n\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Concise, Single Paragraph: Begin with a single paragraph that describes the scene, focusing on key actions in sequence.\n3. Detailed Actions and Appearance: Clearly detail the movements of characters, objects, and relevant elements in the scene. Include brief, essential visual details that highlight distinctive features.\n4. Contextual Setting: Provide minimal yet effective background details that establish time, place, and atmosphere. Keep it relevant to the scene without unnecessary elaboration.\n5. Camera Angles and Movements: Mention camera perspectives or movements that shape the viewer’s experience, but keep it succinct.\n6. Lighting and Color: Incorporate lighting conditions and color tones that set the scene’s mood and complement the actions.\n7. Source Type: Reflect the nature of the footage (e.g., real-life, animation) naturally in the description.\n8. No Additional Commentary: Do not include instructions, reasoning steps, or any extra text outside the described scene. Do not provide explanations or justifications—only the final prompt description.\n\nExample Style:\nA group of colorful hot air balloons take off at dawn in Cappadocia, Turkey. Dozens of balloons in various bright colors and patterns slowly rise into the pink and orange sky. Below them, the unique landscape of Cappadocia unfolds, with its distinctive “fairy chimneys” - tall, cone-shaped rock formations scattered across the valley. The rising sun casts long shadows across the terrain, highlighting the otherworldly topography.\n"
        elif media_type == "image":
            media_description = "You are transforming user inputs into descriptive prompts for generating AI still images. Follow these steps to produce the final description:\n\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Concise, Single Paragraph: Begin with a single paragraph that describes the scene, focusing on key visual elements.\n3. Detailed Appearance: Clearly detail the appearance of characters, objects, and relevant elements in the scene. Include brief, essential visual details that highlight distinctive features.\n4. Contextual Setting: Provide minimal yet effective background details that establish time, place, and atmosphere. Keep it relevant to the scene without unnecessary elaboration.\n5. Composition and Perspective: Mention the composition or perspective that shapes the viewer’s experience, but keep it succinct.\n6. Lighting and Color: Incorporate lighting conditions and color tones that set the scene’s mood and complement the visual elements.\n7. Source Type: Reflect the nature of the image (e.g., real-life, illustration, painting) naturally in the description.\n8. No Additional Commentary: Do not include instructions, reasoning steps, or any extra text outside the described scene. Do not provide explanations or justifications—only the final prompt description.\n\nExample Style:\nA serene mountain lake reflects the towering snow-capped peaks under a clear blue sky. The water is perfectly still, mirroring the jagged cliffs and lush pine forests that surround it. A small wooden dock extends into the lake, with a lone red canoe tied to its edge. The golden light of late afternoon bathes the scene, casting warm hues across the landscape and creating a peaceful, tranquil atmosphere. The composition centers on the lake, with the mountains rising dramatically in the background, framed by the dense greenery on either side.\n"
        elif media_type == "subtle video":
            media_description = "You are transforming user inputs into descriptive prompts for generating AI Videos. Follow these steps to produce the final description:\n\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Concise, Single Paragraph: Begin with a single paragraph that describes the scene, focusing on key actions in sequence.\n3. Detailed Actions and Appearance: Clearly detail the movements of characters, objects, and relevant elements in the scene. Include brief, essential visual details that highlight distinctive features.\n4. Contextual Setting: Provide minimal yet effective background details that establish time, place, and atmosphere. Keep it relevant to the scene without unnecessary elaboration.\n5. Camera Angles and Movements: Mention camera perspectives or movements that shape the viewer’s experience, but keep it succinct.\n6. Lighting and Color: Incorporate lighting conditions and color tones that set the scene’s mood and complement the actions.\n7. Source Type: Reflect the nature of the footage (e.g., real-life, animation) naturally in the description.\n8. No Additional Commentary: Do not include instructions, reasoning steps, or any extra text outside the described scene. Do not provide explanations or justifications—only the final prompt description.\n\nExample Style:\nA group of colorful hot air balloons take off at dawn in Cappadocia, Turkey. Dozens of balloons in various bright colors and patterns slowly rise into the pink and orange sky. Below them, the unique landscape of Cappadocia unfolds, with its distinctive fairy chimneys - tall, cone-shaped rock formations scattered across the valley. The rising sun casts long shadows across the terrain, highlighting the otherworldly topography."
        elif media_type == "image upscale":
            media_description = "You are transforming user inputs into descriptive prompts for generating AI Images. Follow these steps to produce the final description:\n\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Concise, Single Paragraph: Begin with a single paragraph that describes the image, focusing on key visual elements and their arrangement.\n3. Detailed Visual Elements: Clearly describe the objects, characters, or items in the image, emphasizing their visual characteristics, textures, and any distinctive features.\n4. Minimal Context: Provide only essential background details that are explicitly visible in the image, such as the setting or environment. Avoid adding unnecessary context or assumptions.\n5. Composition and Perspective: Mention the arrangement of elements and the perspective or angle of the image, if relevant.\n6. Lighting and Color: Describe the lighting conditions and color tones present in the image, including shadows, highlights, and color gradients.\n7. Source Type: Reflect the nature of the image (e.g., realistic, cartoonish, abstract) naturally in the description.\n8. No Additional Commentary: Do not include instructions, reasoning steps, or any extra text outside the described image. Do not provide explanations or justifications—only the final prompt description.\n\nExample Style:\nA group of colorful hot air balloons take off at dawn in Cappadocia, Turkey. Dozens of balloons in various bright colors and patterns slowly rise into the pink and orange sky. Below them, the unique landscape of Cappadocia unfolds, with its distinctive \"fairy chimneys\" - tall, cone-shaped rock formations scattered across the valley. The rising sun casts long shadows across the terrain, highlighting the otherworldly topography."
        elif media_type == "sto1o":
            media_description = "You are transforming user inputs into descriptive prompts for generating AI Images. Follow these steps to produce the final description:\n\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Concise, Single Paragraph: Begin with a single paragraph that describes the image, focusing on key elements and their arrangement.\n3. Detailed Elements and Appearance: Clearly detail the objects, characters, or items in the image, emphasizing their visual characteristics and any unique features that make one stand out as the \"odd one out.\"\n4. Contextual Setting: Provide minimal yet effective background details that establish the scene's context, such as location, environment, or theme. Keep it relevant without unnecessary elaboration.\n5. Composition and Perspective: Mention the arrangement of elements and the perspective or angle that highlights the \"odd one out\" effectively.\n6. Lighting and Color: Incorporate lighting conditions and color tones that enhance the visual contrast and draw attention to the odd element.\n7. Source Type: Reflect the nature of the image (e.g., realistic, cartoonish, abstract) naturally in the description.\n8. No Additional Commentary: Do not include instructions, reasoning steps, or any extra text outside the described image. Do not provide explanations or justifications—only the final prompt description.\n\nExample Style:\nA vibrant collection of fruits arranged neatly on a wooden table, including apples, oranges, bananas, and a single pineapple. The fruits are brightly colored, with the apples in shades of red and green, the oranges in warm orange tones, and the bananas in yellow. The pineapple stands out with its spiky, textured surface and golden-brown hue, contrasting sharply with the smooth, round shapes of the other fruits. The background features a soft, blurred kitchen setting with warm lighting, creating a cozy atmosphere. The perspective is slightly overhead, emphasizing the arrangement and making the pineapple the clear \"odd one out.\""
        else:
            media_description = "You are transforming user inputs into descriptive prompts for generating AI Videos. Follow these steps to produce the final description:\n1. English Only: The entire output must be written in English with 80-150 words.\n2. Minimal Alterations: Ensure the description closely resembles the input image, making only subtle adjustments that enhance clarity while preserving the original appearance.\n3. Concise, Single Paragraph: Begin with a single paragraph that describes the scene, focusing on key actions in sequence.\n4. Detailed Yet Subtle Adjustments: Clearly describe movements, objects, and elements in the scene, making only minor refinements to highlight essential features.\n5. Contextual Setting: Provide just enough background to establish time, place, and atmosphere without unnecessary elaboration.\n6. Camera Angles and Movements: Mention framing or movement subtly to enhance the scene’s flow.\n7. Lighting and Color: Maintain original tones while refining descriptions to subtly reflect mood and atmosphere.\n8. Source Type: Ensure the description naturally reflects the nature of the footage (e.g., real-life, animation).\n9. No Additional Commentary: Do not include instructions, reasoning steps, or extra text outside the described scene. Only provide the final prompt description.\n\nExample Style:\nA group of colorful hot air balloons take off at dawn in Cappadocia, Turkey. Dozens of balloons in various bright colors and patterns slowly rise into the pink and orange sky. Below them, the unique landscape of Cappadocia unfolds, with its distinctive fairy chimneys – tall, cone-shaped rock formations scattered across the valley. The rising sun casts long shadows across the terrain, highlighting the otherworldly topography.\n"



        # Offload the model if needed
        model.to(offload_device)
        mm.soft_empty_cache()
        ollama = f'{{\n  "instruction": "{image_type_description}",\n  "description": "{final_prompt}"\n}}'

        print(f"Generated prompt: {final_prompt}")
        print(f"Image type description: {image_type_description}")
        print(f"Media description: {media_description}")
        print(f'{{\n  "instruction": "{image_type_description}",\n  "description": "{final_prompt}"\n}}')
        return (final_prompt, image_type_description, media_description, ollama)
