Welcome to my page. I create nodes that I need to make my life easier, most of the stuff I do is based on Image Generation and Manipulation. If I find something lacking I try to create something that helps me or shortens the time required to complete the task at hand. This is by no means an extensive list, more to follow though.

[GR Prompt Selector](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-prompt-selector)

[GR Image Resizer](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-iamge-resizer)

[GR Mask Create](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-mask-create)


If you use these, follow my YouTube channel where I create ComfyUI workflows from scratch

[My YouTube Channel](https://www.youtube.com/channel/UCK4AxxjpeECN4GKojVMqL5g)

Hope this stuff is helpful


# GR Prompt Selector

Can choose from 6 prompts

Always prompt is always on

Negative prompt included

![grpromptselector](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/e74d6aa6-3e5a-4c5a-91c2-3a9a2f65b7b4)


# GR Image Resizer

Resizes image by height and size

Credit to simeonovich

![image](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/da0c6a13-4b08-4798-9333-b7d2e34d6515)

# GR Mask Create

This node creates a single mask that you can use

![image](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/cd82a7d5-1c4e-458c-bdf1-c61b0ec85fad)



# GR Multi Mask Create

If you need multiple mask in one node, you can use this. This node creates upto 8 equal size masks.

![image](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/0b5f684c-40c7-476b-9048-651638f09f83)

# GR Mask Resize

When you need to resize your mask image to fit your latent image

![image](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/26bab87d-add3-43f4-81c3-a64e4e326c0a)

# GR Image Size

A node with preselected image sizes or custom, outputs height and width, can also be used for empty latent image via the latent output

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/c64b42d9-4b28-4c25-95a3-b39fc28911d8)

# GR Tile Image

A node to add a border in pixels and tile the image

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/8a2b4ddc-f69b-40db-9c40-5bf5fe9d1d6d)

# GR Mask Create Random

Creates a random mask of your chosen size, useful to set a latent noise mask at random

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/285f94bd-8ba7-4be8-828b-c61a9a96d3f1)

#GR Tile and Border Image Random Flip

This node takes an image and tiles the image into defined columns and rows, but before outputting it flips one of the tiles randomly

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/88779540-848f-4a37-b5bf-9f0f229e1191)

# GR Stack Image

This node takes two images as input and stacks one on top of the other, both images need to be of the same dimensions

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/1f86b235-832e-497f-a23c-9865538309dc)

# GR Image Resize Methods

This node is a slightly improved Resize node that loads and displays the image (Similar to LoadImage), you can resize the image using different Interpolation methods.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/c6d495ec-ba6e-4ecb-b34e-7706b3a70724)



# Installation:

Install in your custom_nodes directory:

Git clone https://github.com/GraftingRayman/ComfyUI_GraftingRayman
