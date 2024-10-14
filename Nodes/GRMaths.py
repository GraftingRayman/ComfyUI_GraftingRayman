

class GRFloatsNode:


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": ("FLOAT", {"step":0.001}),
                "input_2": ("FLOAT", {"step":0.001}),
                "input_3": ("FLOAT", {"step":0.001}),
                "input_4": ("FLOAT", {"step":0.001}),
                "input_5": ("FLOAT", {"step":0.001}),
                "input_6": ("FLOAT", {"step":0.001}),
                "input_7": ("FLOAT", {"step":0.001}),
            }
        }
    
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "GraftingRayman/Maths"
    FUNCTION = "grfloats"


    def grfloats(self, input_1: float, input_2: float, input_3: float, input_4: float, input_5: float, input_6: float, input_7: float):
        # The input values are replicated or distributed across 21 blocks
        input_list = [input_1, input_2, input_3, input_4, input_5, input_6, input_7]
        output_list = []

        # Strategy 1: Repeating the 7 inputs to fill up 21 blocks
        for i in range(21):
            output_list.append(input_list[i % 7])

        # Alternatively, you could interpolate values or apply a custom transformation

        return (output_list,)  # Return as a tuple since FLOATS is expected to be a single output
