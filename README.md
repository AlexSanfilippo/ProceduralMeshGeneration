# Procedurally Generated Spaceships
A Python program that generates infinite spaceships, vertex-by-vertex. 

![BlenderShips](/GitHubExamples/ships_2by2.png?raw=true)


## Description
Need 3D spaceship meshes for your game or animation?  Need ALOT of them?  Then you've come to the right place.  My procedural generation program uses RNG to build nifty little spaceships.  Use the UI to change the generation parameters and find your favorite ships.  Export the models with a single click and load into your game engine or 3D modeling software of choice.  

![BlenderShips](/GitHubExamples/demo.gif?raw=true)

**Export Ships to .obj format, and open in software like Blender!**
![BlenderShips](/GitHubExamples/BlenderDemoTwoShips.png?raw=true)

## How the ships are generated
OpenGL is essentially a machine that takes in the vertices of triangles and turns them into colored pixels on your screen.  To procedurally generate a model is simply to define these triangles.  

-generate an N-sided regular polygon face  
-calculate the normal of the polygon  
-extrude vertices along the normal to create N new vertices, creating a geometrically similar polygon  
-scale the new polygon  
-connect the two polygons with quadrilaterals to form a closed shape  
-repeat the process on any of the faces  

## How to use:
If you just want to play with the compiled program, download it (for FREE!) on itch.io: 
[[Spaceship Generator 3D](https://ceruleanboolean141.itch.io/spaceship-maker)]


How to run this code:
-pip install the required packages (see below)  
-Run the file MeshWorkshop.py  
**REQUIRED PACKAGES**  
If you would like to run the code for yourself, you will need the following python packages:  
altgraph==0.17.4  
cachetools==5.4.0  
ffmpeg==1.4  
future==1.0.0  
glfw==2.7.0  
importlib_metadata==8.2.0  
multipledispatch==1.0.0  
numpy==1.26.4  
packaging==24.1  
zipp==3.19.2  

# Credits
Skyboxes were sourced from the following cites: 
[[Mountain Lake](https://learnopengl.com/Advanced-OpenGL/Cubemaps)]
[[Starry Sky](https://opengameart.org/content/galaxy-skybox)]
[[Mountain Lake](https://opengameart.org/content/space-nebulas-skybox)]



