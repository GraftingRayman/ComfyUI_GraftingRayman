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

If there is an image without a prompt, it will ask if you want a prompt generated, if yes it will save it with the same filename as the image.

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



# GR Background Remover REMBG

This node is an updated version of other REMBG nodes, includes all available REMBG models to choose from

8 Different models to choose from

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/6fc18a96-86f1-4e19-a89e-e22507fc793f)

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/e9592033-0928-425f-8fd0-2da4c3a539a2)

Use the general use model for all types of images

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/2f0e6f6d-278c-45a3-aea7-9e6d13671fbd)

Use the anime model for anime images

![image](https://github.com/GraftingRayman/ComfyUI_GraftingRayman/assets/156515434/f6760b21-6910-46b4-95b1-7436c4629fd2)
