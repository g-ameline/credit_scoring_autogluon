Hey !

# links
[subject] (https://01.kood.tech/git/root/public/src/branch/master/subjects/ai/credit-scoring)  
[audit] (https://01.kood.tech/git/root/public/src/branch/master/subjects/ai/credit-scoring/audit)  

Those projects are usually developped and presented within notebooks;
you are theorically entitled to request the project be able to run python scripts  
but in practice it will be a bit of a pain for you and me to make it possible.
I rather tried to make is easy for you to run open it as a notebook.

I suggest you 3 different ways to audit the project:
- you can just read the pdf version of each notebook and the saved last output
- you can start a jupyter server and just ***read*** the notebook 
- you can build environment ***within container*** and rerun all the notebooks and reprocess all data   
  - (can take some computing time, depending on your machine) 
- you can build environment ***locally*** and rerun all the notebooks and reprocess all data   
  - (can take some computing time, depending on your machine) 
- you can necessitate me to deliver you proper python script because you want it exactly as described in the instructions
  - notebook can convert themself to script automatically so I can do it if you wish, I do not do it by default because there is enough stuff around already

# about notebooks
this is some sort of interactive scripting page that run in a browser;  
- it allows you to debug and ouput/display intermediate results at any point of your script
- last output carry on in the page if copied (like here) 
- so *you* can visuzalize without rerunning everything
- there should be some pdf of the notebooks if you want to just read it
## if you want to rerun the notebook/scrips yourself:
### and you do **not** want to install all the jupyter stuff lacally(even python) :
go in  the repo/containing/ folder and `bash ./all_launch.sh`
then there is some printing and 
```
 To access the server, open this file in a browser:
        file:///home/mambauser/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://61f1dd9a8829:1234/lab?token=some_random_token_stuff        
        http://127.0.0.1:1234/lab?token=some_random_token_stuff
```
try opening one of the two last links in your browser
then inside the jupyter lab window, from left pane, 
- go to notebooks  
- double click on titanic.ipynb 
- pick up the kernel *kernel_in_container"
- and then click on double arrow >> (restart kernel and run all cells)

### you want to run it locally :
- first, you will need to duplicate the virtual environment (see below)
- install kernel and stuff so that the jupyter server can run notebooks from *that* virtual environment
- get jupyter started in that repo
- run all cells in descending order (run tab then run all or shift+enter on each cells)

# duplicaing vrtual environment
this means installing ***locally*** the versionned language, packages and dependencies.
depending what you are using as package manager :  
- conda
- pip
- mamba
- other

you might want to recreate a virtual environment from :  
- environment.yml or requirement_conda.txt
  - if using conda
- requirements_pip.txt 
  - if using pip
- environment_cross.yml
  - if using macos and conda

*that's mostly stuff I read from the net, I could not test exporting my virtual environment to other package manager than conda.  
If have any issue, please tell me.*

## author
gameline
