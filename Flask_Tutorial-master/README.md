Click on `overview.ipynb` for brief overview of a basic Flask framework.

To launch template,
1. Download repo
2. Ensure Flask is installed (See <a href="http://flask.pocoo.org/docs/0.12/installation/">install documentation)</a><br>
    <code> pip install Flask </code>
3. Navigate to repo in terminal / command line
4. Navigate to file_hierarchy/app_name
5. Launch app <br>
    <code> python run.py </code>
6. Open your browser and navigate to either <br>
    a. localhost:5000 <br>
    b. 127.0.0.1:5000

## Comments
In the beginning, the main place you will be doing your development is in the following files:<br><br>
a. `app_name/app_name/views.py` <br>
    Here is where the interfacing between JS and Python happens. When you setup your AJAX calls, this is the file those server-side methods go in.<br><br>
b. `app_name/app_name/templates/index.html` <br>
    Your basic index.html is here. More advanced templating techniques exist. For my purposes, I build everything dynamically in `D3.js`, and my index.html only consists of css, js references and the `<body></body>` tag to which I append everything via D3. <br><br>
c. `app_name/app_name/static/js/script.js` <br>
    All your javascript goes in here. If you add javascript files to this directory, make sure to reference to them in the index.html template. <br><br>
d. `app_name/app_name/static/css/style.css` <br>
    All your CSS goes in here. If you add CSS files to this directory, make sure to reference to them in the index.html template. <br>

## Flask Project Examples
Below are two example projects built with Flask with varying levels of complexity. May be useful for familiarizing yourself with file structures, how to make AJAX calls, D3, etc. <br>

1. Artificial Neural Network Parameter Visualizer: <a href="https://github.com/michael-iuzzolino/ann_hinton_visualizer">ANN Hinton Visualizer</a>
2. Musical App for HRI project: <a href="https://github.com/michael-iuzzolino/hri_music_puzzle_task">HRI Music Puzzle Task</a>
