# Procedurally Generated Spaceships
![BlenderShips](/GitHubExamples/BlenderDemoTwoShips.png?raw=true)

(WHAT IS THIS ( ))
Need 3D spaceship meshes for your game or animation?  Need ALOT of them?  Then you've come to the right place.  My procedural generation program uses RNG to build nifty little spaceships.  Use the UI to change the generation parameters and find your favorite ships.  Export the models with a single click and load into your game engine or 3D modeling software of choice.  

**Export Ships to .obj format, and open in software like Blender!**
![BlenderShips](/GitHubExamples/BlenderDemoTwoShips.png?raw=true)

(How does it work)
(general)
  -Uses PyOpenGL
  -All code in python, except for shaders, which are GLSL
Procedural generation system 
  -Origin->face
  -Triangulize the face
  -Extrude the face
  -Switch two similar faces

(How to use it)
If you just want to play with the compiled program, check it out (for FREE!) on itch.io: 
[ITCH LINK TO COME]

If you would like to run the code for yourself, you will need the following python packages

How to run this code:
-pip install the required packages (see below)\n
-Run the file MeshWorkshop.py\n
**REQUIRED PACKAGES**\n
altgraph==0.17.4\n
cachetools==5.4.0\n
ffmpeg==1.4\n
future==1.0.0
glfw==2.7.0
importlib_metadata==8.2.0
multipledispatch==1.0.0
numpy==1.26.4
packaging==24.1
zipp==3.19.2
