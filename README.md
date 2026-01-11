Welcome to my page. I create nodes that I need to make my life easier, most of the stuff I do is based on Image Generation and Manipulation. If I find something lacking I try to create something that helps me or shortens the time required to complete the task at hand. This is by no means an extensive list, more to follow though.

If you would like to make a donation to my efforts use the QR code

![QR code](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/a65f6614-94d3-46e2-8d4e-f60e98d44d6f)


# Installation:

Install using ComfyUI Manager or manually install in your custom_nodes directory with the following command:

"git clone https://github.com/GraftingRayman/ComfyUI_GraftingRayman"

**Make sure to install Clip**

For comfyui portable run the following command in your comyfui folder

 **".\python_embeded\python.exe -m pip install git+https://github.com/openai/CLIP.git"**

 For system python run the following command

 **"pip install git+https://github.com/openai/CLIP.git"**

 Without this the nodes will fail to be imported


If you use these, follow my YouTube channel where I create ComfyUI workflows from scratch

[![Youtube Badge](https://img.shields.io/badge/Youtube-FF0000?style=for-the-badge&logo=Youtube&logoColor=white&link=https://www.youtube.com/channel/UCK4AxxjpeECN4GKojVMqL5g)](https://www.youtube.com/channel/UCK4AxxjpeECN4GKojVMqL5g)

Hope this stuff is helpful


# GR Prompt Viewer

<b>UPDATE:</b>
If the prompt has an image with the same name, it will now display the image below the prompt.

A node to save, edit and load your prompts, never lose a prompt. View dynamically the contents before using the prompt.

<img width="1936" height="1513" alt="image" src="https://github.com/user-attachments/assets/7f559cd3-0a2f-4e9e-be22-538298df989e" />

All prompts are saved in the node directory inside a folder called prompts comfyui/custom_nodes/ComfyUI_GraftingRayman/Nodes/prompts

Prompts can be sorted by folders

All executed prompts are autosaved when it is queued

<img width="1108" height="454" alt="image" src="https://github.com/user-attachments/assets/e3e8b456-a8fe-46df-9250-ef6972885997" />

<img width="1164" height="528" alt="image" src="https://github.com/user-attachments/assets/0cb21e54-7b89-4e3a-bebf-33cabccb383c" />


# GR Menu Hook

This node adds some features that are not needed, but they are wanted

Keeps nodes inside a group, can arrange them in multiple ways.

Also can add all nodes that are not in groups into a node with one click

<img width="404" height="151" alt="image" src="https://github.com/user-attachments/assets/b9548267-1409-46cf-98a1-85c385c206aa" />

<img width="395" height="143" alt="image" src="https://github.com/user-attachments/assets/bec26850-a0f3-4a2e-84c9-e8b2058aab31" />

<img width="1530" height="895" alt="image" src="https://github.com/user-attachments/assets/41139646-3251-4169-9e94-e6e37bed610d" />

<img width="1308" height="901" alt="image" src="https://github.com/user-attachments/assets/ba757b77-1ed2-4159-9381-6c8575b7526b" />

<img width="1302" height="889" alt="image" src="https://github.com/user-attachments/assets/c1c2848b-1d2a-4ec2-97f0-91d62bf195f8" />








# GR Lora Loader

A LoRA loader node, that dynamically updates as LoRA's are added to the node.
LoRA's can be randomized along with their weights or done manually

<img width="610" height="511" alt="image" src="https://github.com/user-attachments/assets/111b2575-73a9-48e6-986f-dd9596f55b08" />
<img width="606" height="540" alt="image" src="https://github.com/user-attachments/assets/887b5efc-0485-41c9-851d-547142b504f6" />


# GR Florence 2 Caption Generator for Ollama

![image](https://github.com/user-attachments/assets/0ab41137-ba8a-46d7-8a7f-7ec2700384b5)


# GR Prompt Generator

If like me, your prompts are very bland and can't seem to get the right amount of details in, this node is for you. This node generates the positive prompt for scenes. You can add your own details to be appended to the prompt as well. Short text can be used to describe the image, and this can be then expanded automatically. A vast array of the type of image can be selected, images can also be replicated using the seed.

![image](https://github.com/user-attachments/assets/206b6a30-d4ae-4ed8-95e7-e3dce8ffa4fd)
![image](https://github.com/user-attachments/assets/06af39bc-0214-4b8c-b4fb-d29b001f1198)
![image](https://github.com/user-attachments/assets/700c45f3-fb84-4842-999f-228ab2aa79e2)

There is a new Extended version of the Prompt Generator with over 10 Quintillion+ combinations for prompts. You can choose from more than 120 different subjects to generate prompts


# GR Pan Or Zoom
Workflow in workflow folder.

The GR Pan Or Zoom node, takes an image of your choice and using the depth map can zoom or pan the image using 6 different depth focusing methods.

![image](https://github.com/user-attachments/assets/5ae49167-c4cb-4a67-bc42-e0b4bc8ce123)


With out Depth

![AnimateDiff_00136](https://github.com/user-attachments/assets/cf913d65-7c9e-4f2b-8621-7147541df8af)

With Depth

![AnimateDiff_00137](https://github.com/user-attachments/assets/c6353fdc-6f71-45a5-85ef-64e09c38f49e)



# GR Text Overlay

This node creates a text overlay, this can be single or multi line. Placement includes left, right, center, middle, top, bottom. Justification can be manually set. Lots of default colours selectable from the list. All your system TTF and OTF fonts available dynamically. Line spacing as well as letter spacing can be controlled in steps of 0.01. Text without stroke thickness can be used as a mask. 
A second mask includes the stroke thickness
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

# GR Image Multiplier

Created this because VHS VideoCombine does not go below 1, what if we wanted each frame to be 0.5 seconds long, then incorporate this node in between your image output and the vhs video combine input and voila!  you have 0.5 if the multiplier is set to 2 in this node. Added interleave and randomize just incase anyone wants it

![image](https://github.com/user-attachments/assets/b53e75f2-0aaa-470f-a9da-d9231738eb80)


# GR Image Resizer

Resizes image by height, size, divisible and scale

Credit to simeonovich

![image](https://github.com/user-attachments/assets/b4812b88-be01-44b9-81ed-274550fccd9b)


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

This node takes two images as input and stacks one on top of the other, both images need to be of the same dimensions. Added a new input to select horizontal or vertical stacking.

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

This node, takes a background image (Dimensions of background image are used to rescale the overlay and mask if different), an overlay image and a mask. The overlay image is pasted over the background image with guidance from the mask. The opacity can be manually set along with manual positioning of the overlay image as well as the mask. If you want to see where the mask is being placed you can enable an outline of the mask with configurable outline thickness, colour and opacity along with outline placement. Various blend options are available for the blending of the masked image. 4 Outputs are provied, 1st output sends the overlayed image, 2nd output sends the same overlayed image with the mask inverted, 3rd output sends the outline only and the fourth output sends the dimensions of the background image in text format.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/d1f7451a-fbe9-4cf9-95b6-17ecf114dc05)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/b85abee7-abd9-4ddd-8676-03c1d7f49589)

# GR Counter

This node creates a countdown or count up timer video, you can choose the font and size you want, also includes an outline feature. The node outputs the frames which can be further processed if required. Video is saved in the location specified in the node which is of type MP4. Can move the timer across the screen or resize the timer from start to finish.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/99df7bac-2331-41df-97e5-ca34983e3bfc)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/2a23097f-9db7-463f-af00-b6167beccd08)

![AnimateDiff_00063](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/d7fdfee0-6c6f-498c-bb3e-11ab8c8c0d7e)

# GR Background Remover REMBG

This node is an updated version of other REMBG nodes, includes all available REMBG models to choose from

8 Different models to choose from

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/6fc18a96-86f1-4e19-a89e-e22507fc793f)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/e9592033-0928-425f-8fd0-2da4c3a539a2)

Use the general use model for all types of images

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/2f0e6f6d-278c-45a3-aea7-9e6d13671fbd)

Use the anime model for anime images

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/f6760b21-6910-46b4-95b1-7436c4629fd2)

# GR Scroller

A node that creates a scrolling text on a background colour of your choice. You can select the path it saves the video or individual frames.

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/4f1fe40a-f4d1-465c-8c35-00293682b881)

Lots of different styles to select from

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/d29b328b-3417-462b-8f0c-b945428b48fa)

