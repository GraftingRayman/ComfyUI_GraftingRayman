import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from PIL import Image
import os
import comfy.model_management as mm
import hashlib
import gc

class GRPromptGenerator:
    # Class-level cache for models
    _model_cache = {}
    _processor_cache = {}
    _current_model_key = None  # Track which model is currently loaded
    
    # Model type detection
    MODEL_TYPES = {
        'qwen2vl': [
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-3B-Instruct',
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-72B-Instruct',
        ],
        'moondream': [
            'vikhyatk/moondream2',
            'vikhyatk/moondream1',
        ],
        'smolvlm': [
            'HuggingFaceTB/SmolVLM-Instruct',
            'HuggingFaceTB/SmolVLM-256M-Instruct',
            'HuggingFaceTB/SmolVLM-500M-Instruct',
        ],
    }
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Flatten all models into one list
        all_models = []
        for model_list in cls.MODEL_TYPES.values():
            all_models.extend(model_list)
        
        return {
            "required": {
                "question": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "pre_text": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "model": (all_models, {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),  # Option to cache or not
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "GraftingRayman/Image Processing"

    def hash_seed(self, seed):
        """Hash the seed for reproducibility."""
        seed_bytes = str(seed).encode('utf-8')
        hash_object = hashlib.sha256(seed_bytes)
        return int(hash_object.hexdigest(), 16) % (2**32)

    def get_model_type(self, model_path):
        """Detect which model type we're using."""
        for model_type, model_list in self.MODEL_TYPES.items():
            if model_path in model_list:
                return model_type
        # Default to qwen2vl for unknown models
        return 'qwen2vl'

    @classmethod
    def clear_cache(cls):
        """Clear all cached models from memory."""
        print("[UnifiedVisionPromptGenerator] Clearing model cache...")
        
        # Move models to CPU and delete
        for key, model in cls._model_cache.items():
            try:
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            except Exception as e:
                print(f"Error clearing model {key}: {e}")
        
        cls._model_cache.clear()
        cls._processor_cache.clear()
        cls._current_model_key = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[UnifiedVisionPromptGenerator] Cache cleared.")

    @classmethod
    def unload_if_different(cls, new_cache_key):
        """Unload current model if switching to a different one."""
        if cls._current_model_key and cls._current_model_key != new_cache_key:
            print(f"[UnifiedVisionPromptGenerator] Switching models, unloading {cls._current_model_key}")
            
            # Unload the old model
            if cls._current_model_key in cls._model_cache:
                old_model = cls._model_cache[cls._current_model_key]
                try:
                    if hasattr(old_model, 'to'):
                        old_model.to('cpu')
                    del cls._model_cache[cls._current_model_key]
                    del cls._processor_cache[cls._current_model_key]
                except Exception as e:
                    print(f"Error unloading old model: {e}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_path, device, keep_loaded=False):
        """Load the processor and model with optional caching."""
        try:
            model_type = self.get_model_type(model_path)
            cache_key = f"{model_path}_{model_type}"
            
            # If not keeping models loaded, clear cache before loading new model
            if not keep_loaded:
                self.clear_cache()
            else:
                # If keeping loaded but switching models, unload the old one
                self.unload_if_different(cache_key)
            
            # Check cache first
            if cache_key in self._model_cache and cache_key in self._processor_cache:
                print(f"[UnifiedVisionPromptGenerator] Using cached model: {model_path}")
                model = self._model_cache[cache_key]
                processor = self._processor_cache[cache_key]
                
                # Ensure model is on correct device
                if hasattr(model, 'parameters') and next(model.parameters()).device != device:
                    print(f"[UnifiedVisionPromptGenerator] Moving cached model to {device}")
                    model = model.to(device)
                    self._model_cache[cache_key] = model
                
                self._current_model_key = cache_key
                return processor, model, model_type
            
            print(f"[UnifiedVisionPromptGenerator] Loading model: {model_path}")
            
            # Load based on model type
            if model_type == 'qwen2vl':
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    from qwen_vl_utils import process_vision_info
                except ImportError:
                    print("Error: Qwen2VL requires 'qwen-vl-utils'. Install with: pip install qwen-vl-utils")
                    raise
                    
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(device)
                
            elif model_type == 'moondream':
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(device)
                processor = tokenizer
                
            else:  # smolvlm
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                    ).to(device)
                except:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    ).to(device)
            
            # Only cache if keep_loaded is True
            if keep_loaded:
                self._model_cache[cache_key] = model
                self._processor_cache[cache_key] = processor
                self._current_model_key = cache_key
                print(f"[UnifiedVisionPromptGenerator] Model cached in VRAM")
            else:
                print(f"[UnifiedVisionPromptGenerator] Model loaded (not cached)")
            
            return processor, model, model_type
            
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            raise

    def convert_image(self, image):
        """Safely convert image tensor to PIL Image."""
        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                elif image.dim() != 3:
                    raise ValueError(f"Unexpected image dimensions: {image.dim()}")
                
                if image.is_cuda:
                    image = image.cpu()
                
                image_np = image.mul(255).clamp(0, 255).byte().numpy()
                
                if image_np.shape[0] in [1, 3, 4]:
                    image_np = image_np.transpose(1, 2, 0)
                
                if image_np.shape[2] == 1:
                    image_np = image_np.repeat(3, axis=2)
                elif image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]
                
                return Image.fromarray(image_np)
            elif isinstance(image, Image.Image):
                return image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            print(f"Error converting image: {str(e)}")
            raise

    def generate_qwen2vl(self, processor, model, device, question, image, max_new_tokens, temperature):
        """Generate using Qwen2-VL models."""
        from qwen_vl_utils import process_vision_info
        
        if image is not None:
            pil_image = self.convert_image(image)
            temp_path = "/tmp/temp_qwen_image.png"
            pil_image.save(temp_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_path}"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def generate_moondream(self, processor, model, device, question, image, max_new_tokens):
        """Generate using Moondream models."""
        if image is None:
            return "Moondream requires an image input."
        
        pil_image = self.convert_image(image)
        enc_image = model.encode_image(pil_image)
        
        with torch.no_grad():
            try:
                answer = model.answer_question(enc_image, question, processor, max_tokens=max_new_tokens)
            except:
                answer = model.answer_question(enc_image, question, processor)
        
        # Clean up the answer - remove question if it's echoed
        answer = answer.strip()
        
        patterns_to_remove = [
            question,
            f"{question}:",
            f"{question} ",
            f"Question: {question}",
            f"Q: {question}",
        ]
        
        for pattern in patterns_to_remove:
            if answer.lower().startswith(pattern.lower()):
                answer = answer[len(pattern):].strip()
                answer = answer.lstrip(':-').strip()
                break
        
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        elif answer.lower().startswith("a:"):
            answer = answer[2:].strip()
        
        return answer

    def generate_smolvlm(self, processor, model, device, question, image, max_new_tokens, temperature):
        """Generate using SmolVLM models."""
        if image is not None:
            pil_image = self.convert_image(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_text, images=[pil_image], return_tensors="pt").to(device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]
        
        return output_text.strip()

    def generate_prompt(self, question, pre_text, max_new_tokens, temperature, seed, model, keep_model_loaded=False, image=None):
        """Generate a prompt using the selected vision model."""
        generated_ids = None
        inputs = None
        pil_image = None
        model_obj = None
        should_unload = not keep_model_loaded
        
        try:
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()

            processor, model_obj, model_type = self.load_model(model, device, keep_model_loaded)

            if seed:
                torch.manual_seed(self.hash_seed(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.hash_seed(seed))

            # Generate based on model type
            if model_type == 'qwen2vl':
                generated_prompt = self.generate_qwen2vl(processor, model_obj, device, question, image, max_new_tokens, temperature)
                
            elif model_type == 'moondream':
                generated_prompt = self.generate_moondream(processor, model_obj, device, question, image, max_new_tokens)
                
            elif model_type == 'smolvlm':
                generated_prompt = self.generate_smolvlm(processor, model_obj, device, question, image, max_new_tokens, temperature)
            
            else:
                generated_prompt = "Unknown model type."

            # Add pre_text to the beginning of the generated prompt if provided
            if pre_text and pre_text.strip():
                generated_prompt = f"{pre_text.strip()}. {generated_prompt}"

            print(f"Model Type: {model_type}")
            print(f"Generated prompt: {generated_prompt}")
            
            return (generated_prompt,)
            
        except Exception as e:
            print(f"Error in generate_prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("",)
            
        finally:
            # Clean up intermediate tensors
            if generated_ids is not None:
                del generated_ids
            if inputs is not None:
                del inputs
            if pil_image is not None:
                del pil_image
            
            # If not keeping model loaded, unload it
            if should_unload and model_obj is not None:
                try:
                    print("[UnifiedVisionPromptGenerator] Unloading model after inference")
                    if hasattr(model_obj, 'to'):
                        model_obj.to('cpu')
                    self.clear_cache()
                except Exception as e:
                    print(f"Error during model cleanup: {e}")
            
            # Always clean up ComfyUI memory
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

