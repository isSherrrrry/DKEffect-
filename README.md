# TO RUN THE CODE: 

Go into the 'categorization_experiment' directory

1. In the terminal, navigate to the project directory. 
	- If you have Python 3 as your main python version:
	
		python -m http.server 3000

	- If you have Python 2 as your main python version:
	
		python -m SimpleHTTPServer 3000

2. In the web browser, navigate to localhost:3000/consent.html


# SUMMARY 

The code is organized across a series of html pages, which pass relevant information to one another via local storage in the browser. 

Below is a brief guide to the primary files needed to branch an experiment from this codebase. 


# FILE CONTENTS

There are other files that are not described below, many of which are related to post-experiment user data and analysis (and should actually be removed from this directory at some point) and others that you will find are no longer relevant in the flow of the study. 

## HTML

consent.html: consent form. Will need to be updated with information from the relevant experiment's IRB-approved consent form.

instructions.html: a brief summary of the components of the experiment to come.  

presurvey.html: an embedded google survey to ask questions before users begin the experiment. Will need to be replaced with links to surveys relevant to the current experiment. 

demo.html: an embedded video demonstrating how to use the interface for the study. Will need to be updated according to any new changes that are made to the interface.

practice.html: the interface loaded with a different and smaller (cars) dataset, intended to give users an opportunity to learn the buttons and functionality of the system (this will reduce noisy interaction data due to learning during the actual experiment). Primary js code is located in js/main_practice.js.

bball_refresher.html: an embedded video that describes some information relevant to the task of categorizing basketball players. Will need to be updated / replaced according to which data we use. 

study.html: the primary page of the study. This is where the main task takes place. Primary js code is located in js/main_bball.js.

postsurvey.html: an embedded google survey to ask questions after users complete the experiment. Will need to be replaced with links to surveys relevant to the current experiment. Also the final page in the study workflow.

## JS

Depending on the phase of the study, the primary scripts are either js/main_practice.js or js/main_cars.js. These files may reference functions in js/scatter-plot_practice.js, js/axis.js, and js/para-coord.js. 

## DATA

practice.csv: a small subset of the cars dataset used in the practice phase to help users get acquainted with the data. 

bball_top100_decimal.csv: the dataset used in the primary study, although there are several variations also contained within the data directory.



# LOGGING

Interactions are logged using lib/ial.js internally and then to an external server using LogEntries. Look for the inclusion of le.min.js in the head of the html file, and look for LE.init() and LE.log() within the javascript. This will need to be updated for each new experiment. Create an account at https://logentries.com to get started. Relevant documentation is here: https://docs.logentries.com/docs/javascript. To create a new token, create a new log set by selecting the javascript library option. Give it a name, then use the generated log token as the argument to LE.init(). 

Note: the free version of LogEntries has limitations on how much data is stored. Be mindful of this for data-intensive experiments. Likewise, it has time limits for how long data is stored. Be sure to download local copies or push them elsewhere before they are cleared from the system. 

There are other options for logging (e.g., Firebase) if you prefer to integrate that instead. 


# TO MOVE THE EXPERIMENT ONLINE 

The original implementation of this code was run locally for in-lab experiments where a study investigator was present and observing. For future experiments that may be run online, there are a number of changes that will likely need to happen to ensure robustness. E.g., 

1. Prevent users from hitting the back button in the browser (and / or utilize a single web-page with hidden divs to contain each "page" of info to reduce the need for users to go back). 

2. Other pages will require validation to ensure complete data (e.g., the survey pages currently do not validate that the form was submitted before continuing to the next page; the demonstration video does not validate that the user has played the video to completion; etc).  

3. More instructions will be required on each page to explain what users should do if a study administrator will not be present. 

4. Add a separate "Thanks" screen at the end.