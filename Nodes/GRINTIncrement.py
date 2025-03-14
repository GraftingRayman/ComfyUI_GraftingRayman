import comfy
import torch

class GRINTIncrement:
    def __init__(self):
        self.counter = 0
        self.last_seed = None  # Track the last seed value

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "", "multiline": True}),  # Add prefix input (multiline string)
                "start_value": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # Seed input
            },
        }

    RETURN_TYPES = ("INT", "STRING")  # Return both INT and STRING
    FUNCTION = "increment"

    CATEGORY = "custom"

    def increment(self, prefix, start_value, seed):
        # Check if the seed has changed
        if seed != self.last_seed:
            self.last_seed = seed  # Update the last seed
            if self.counter == 0:
                self.counter = start_value
            else:
                self.counter += 1  # Increment the counter
        
        # Format the counter as prefix/000014/ (prefix + 6-digit number with slashes)
        formatted_counter = f"{prefix}/{self.counter:06d}/"
        
        return (self.counter, formatted_counter)  # Return both the integer and the formatted string

