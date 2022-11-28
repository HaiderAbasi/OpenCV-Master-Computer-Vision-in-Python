## Windows 10 

1) **Download and install** 
   - **Python 3.8.10** [> Link <](https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe)
      - Install with default settings
      - Select Add python to Path 
      - Test by going to cmd prmpt and write python 
   - **VScode** [> Link <](https://code.visualstudio.com/download)
      - Install with default settings 
2) **Open VScode and Clone Repository** [> Link <](https://github.com/HaiderAbasi/OPENCV4-Master-Computer-Vision-in-Python-)
   - Sign in to your Github account if asked. **(Github Account Required)**
   - Download and Extract Data folder inside Repo Directory [> Link <](https://drive.google.com/drive/folders/1upBrCt0fF7kO_U6hizRqc1DHp7XdEolU?fbclid=IwAR3XPDs70f47MSJn2FybEkC-SVlBF5BSkYFNwUAmbspbMSqvXEK732NUrp4)
 

3) **Configure Vscode**
    > You wil need to setup git on your computer for commiting, pushing etc. 
    - Setup git and link to VSCODE by following this [GUIDE](https://www.geeksforgeeks.org/how-to-install-git-in-vs-code/) **(Step 2 Onwards)**
    - Once done, Install Required and Optional Extension for VScode
       - Required: Python
       - Optional: GitGraph + vscode-icons + atom one dark theme  
    - Then, Setup Python Path for Integrated terminal to point to the Workspace Directory.
    
      > Done to solve Import errors.
       - Press Ctrl + Shift + P to open Command Palette 
       - Go to Users.setting.json 
       - Add the following line
       ``` 
       "terminal.integrated.env.windows": { "PYTHONPATH": "${workspaceFolder}" }
       ```
       - Restart VScode 


4) **Setup Virtual Environment** 

    - Open Cmd Prompt and navigate to repo directory, 
    - Start by adding a new module virtualenv using pip.
      ```
      python -m pip install --user virtualenv
      ```
    > This will be used to create virtual environments 
    - Create a virtual enivonment named venv_3.8.10 
      ```
      python -m virtualenv MCV_Venv 
      ```
    - Activate Virtual envronment 
    - Install required libraries using pip.
      ```
      pip install –r requirements.txt 
      ``` 
    - Change python in vscode to point to Virtual Environment Python.
    
      - Perform change by clicking on python in bottom right corner of Vscode. 
      
    - Test by running a_image_video_access.py to see if it works
    
    > Congratulations, You are now all set to follow the course. :)
