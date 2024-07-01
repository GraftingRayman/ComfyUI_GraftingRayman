Welcome to my page. I create nodes that I need to make my life easier, most of the stuff I do is based on Image Generation and Manipulation. If I find something lacking I try to create something that helps me or shortens the time required to complete the task at hand. This is by no means an extensive list, more to follow though.

# Installation:

Install using ComfyUI Manager or manually install in your custom_nodes directory with the following command:

"git clone https://github.com/GraftingRayman/ComfyUI_GraftingRayman"

**Make sure to install Clip**

For comfyui portable run the following command in your comyfui folder

 **".\python_embeded\python.exe -m pip install git+https://github.com/openai/CLIP.git"**

 For system python run the following command

 **"pip install git+https://github.com/openai/CLIP.git"**

 Without this the nodes will fail to be imported



# Overlays:
[GR Text Overlay](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-text-overlay)

[GR Onomatopoeia](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-onomatopoeia)

# Prompts:
[GR Prompt Selector](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-prompt-selector)

[GR Prompt Selector Multi](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-prompt-selector-multi)

[GR Prompt Hub](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-prompt-hub)

# Image Utils:
[GR Image Resizer](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-iamge-resizer)

[GR Image Size](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-size)

[GR Stack Image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-stack-image)

[GR Image Resize Methods](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-resize-methods)

[GR Image Details](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-details)

[GR Image Paste](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-paste)

[GR Image Paste With Mask](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-paste-with-mask)

# Mask Utils:
[GR Mask Create](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-mask-create)

[GR Multi Mask Create](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-multi-mask-create)

[GR Mask Resize](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-mask-resize)

[GR Image/Depth Mask](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-image-depth-mask)

[GR Mask Create Random](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-mask-create-random)

# Image Tiling
[GR Flip Tile Random Red Ring](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-flip-tile-random-red-ring)

[GR Flip Tile Random Inverted](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-flip-tile-random-inverted)

[GR Tile Image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-tile-image)

[GR Tile and Border Image Random Flip](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/blob/main/README.md#gr-tile-and-border-image-random-flip)


If you use these, follow my YouTube channel where I create ComfyUI workflows from scratch

[![Youtube Badge](https://img.shields.io/badge/Youtube-FF0000?style=for-the-badge&logo=Youtube&logoColor=white&link=https://www.youtube.com/channel/UCK4AxxjpeECN4GKojVMqL5g)](https://www.youtube.com/channel/UCK4AxxjpeECN4GKojVMqL5g)

Hope this stuff is helpful

# GR Text Overlay

This node creates an text overlay, this can be single or multi line. Placement includes left, right, center, middle, top, bottom. Justification can be manually set. Lots of default colours selectable from the list. All your system TTF and OTF fonts available dynamically. Line spacing as well as letter spacing can be controlled in steps of 0.01. Text without stroke thickness can be used as a mask.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/4bcd7c03-a53e-48fa-b7b3-fe7341e6ba83)

A second mask includes the stroke thickness

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/261a9b47-18b6-4b17-9b23-075b2b2303e1)

Added a background box that can be used to highlight the text, the edge styles for the box can be selected as well as its opacity

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/4d3f5cfa-bfd7-418b-8f74-40600b3f35d7)


# GR Onomatopoeia

Creates random Onomatopoeia or uses the letters provided. Still a bit buggy, currently placing it left or right does not work correctly.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/10dc92f6-4941-43ba-b122-e14f79242b9b)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/3049f65d-9e65-44a2-9120-37e272b601e8)


# GR Prompt Selector

Can choose from 6 prompts

Always prompt is always on

Negative prompt included

![grpromptselector](https://github.com/GraftingRayman/ComfyUI_GR_PromptSelector/assets/156515434/e74d6aa6-3e5a-4c5a-91c2-3a9a2f65b7b4)

# GR Prompt Selector Multi

All the features of Prompt Selector, this time you can use all 6 prompt styles at the same time

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/4fa4d338-68cb-48cf-b470-ae34779abbce)

# GR Prompt Hub

This is a simple prompt combiner, will take 6 positives and 6 negatives and combine them

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/e1815752-a20d-4505-b6f0-35bdb6d2b2b6)


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

# GR Image Depth Mask

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/2b24166f-79a2-429e-8c12-77abd5be33cb)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/c17c21a6-a36f-4a68-97c3-4ef5677ddde0)
![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/7598fac0-a339-42ab-bd04-a72ceb026476)
![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/f3ad7848-92ad-48fd-8648-7f1fbd32c8de)
![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/04ad98be-86c1-47ef-a334-eaa2844bd3be)

# GR Image Size

A node with preselected image sizes or custom, outputs height and width, can also be used for empty latent image via the latent output

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/20942df2-f33a-451b-a95b-e3772b26801e)
![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/94d8427b-342c-4cf6-9531-0eb1c1eefc8e)

Added a dimensions input to the node, this takes the dimensions from the image and passes that for the height/width.
A seed feature has also been added with 15 digit random seeds, this will reduce number of nodes on display

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/56b1fc0d-182d-4ccb-9258-c56ac7308b82)

# GR Tile Image

A node to add a border in pixels and tile the image

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/8a2b4ddc-f69b-40db-9c40-5bf5fe9d1d6d)

# GR Mask Create Random

Creates a random mask of your chosen size, useful to set a latent noise mask at random

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/285f94bd-8ba7-4be8-828b-c61a9a96d3f1)

# GR Tile and Border Image Random Flip

This node takes an image and tiles the image into defined columns and rows, but before outputting it flips one of the tiles randomly

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/88779540-848f-4a37-b5bf-9f0f229e1191)

# GR Stack Image

This node takes two images as input and stacks one on top of the other, both images need to be of the same dimensions

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/1f86b235-832e-497f-a23c-9865538309dc)

# GR Image Resize Methods

This node is a slightly improved Resize node that loads and displays the image (Similar to LoadImage), you can resize the image using different Interpolation methods.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/c6d495ec-ba6e-4ecb-b34e-7706b3a70724)

# GR Image Details

I took the standard Image Preview node and updated it, now you can see the preview with additional details like Filename, File Size, Height, Width, Type of Image etc
You can also save the file with the details with the save node

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/cc597614-e8d0-4bf4-b068-4aace58b06c2)
![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/ba5079e1-0219-44e2-b424-467d9411b005)

# GR Flip Tile Random Red Ring

This node takes an image input and creates a tile of the required size, it then flips a random tile and puts a red ring around it

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/e34b39e4-2618-43a8-82b8-ce77435a02a3)

If a second image is provided (this iamge needs to be resized before joining to the dimenions of the main tile + 2 x border size, this will then use this second image as the flipped random tile and put a red ring around it

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/ca1db877-f4cc-4203-a80a-885b0bcae6aa)

For both types, the first output will give you the image with the randomly flipped image, the second output will give you the same image but with the red ring around the flipped tile

# GR Flip Tile Random Inverted

This node takes an image, creates a tile of the required size with a randomly flipped tile, the first output shows the full tile with the randomly flipped tile, while the second output shows the randomly flipped tile inverted.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/5bb0ecfc-5f74-4055-bff2-6821f5ffd7aa)

# GR Checkered Board

Creates a checkered board, outputs an image as well as a mask. You can choose from a myriad of colours, borders for individual boxes and the whole board.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/a7990eb6-cb84-4d2c-928d-62f57546c3a3)

# GR Image Paste

This node takes the second image and pastes it over the first. Opacity can be set for the second image. Both images need to be of the same dimensions, this can be resized using GR Image Resize node.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/06aaa5d4-275c-4abe-b4cc-107f334a124b)

# GR Image Paste With Mask

This node, takes a background image, an overlay image and a mask. The overlay image is pasted over the background image with guidance from the mask. The opacity can be manually set along with manual positioning of the overlay image as well as the mask. If you want to see where the mask is being placed you can enable an outline of the mask with configurable outline thickness, colour and opacity along with outline placement. Various blend options are available for the blending of the masked image. 4 Outputs are provied, 1st output sends the overlayed image, 2nd output sends the same overlayed image with the mask inverted, 3rd output sends the outline only and the fourth output sends the dimensions of the background image in text format.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/d1f7451a-fbe9-4cf9-95b6-17ecf114dc05)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/f7d0d4bc-5682-4dba-9b7b-dd183e08f20b)
