# ml_plugin_service.py
Launches a local REST service, serving as an API bridge to the text detector.<br>

<b>Launch</b><br>
python ml_plugin_serice.py

<b>REST API reference</b><br>

<b>Ping</b><br>
curl http://localhost:5000 

<b>Submit a processing task</b><br>
curl -i -H "Content-Type: application/json" -X POST -d '{"path": "/my_path/my_dir"}' http://localhost:5000/process

<b>List all tasks</b><br>
curl -X GET http://localhost:5000/tasks

<b>Get a specific task json by ID</b><br>
curl -X GET http://localhost:5000/tasks/my_task_id

<b>Abort a running task</b><br>
curl -X DELETE http://localhost:5000/tasks/my_task_id

<b>Delete a non-running task from registry</b><br>
curl -X DELETE http://localhost:5000/tasks/my_task_id

<b>Task fields</b><br>
{<br>
  "task": {<br>
    # task id <br>
    "id": 1, <br>
    "path": "/my_path/my_dir", <br>    
    "status": #one of "created","running","error","aborting","aborted","completed" <br>    
    #When status is "aborted" or "completed", resultSet will be present.<br>
    "resultSet":[{'infile':f,'outfile':outfil,'text_present':1,"maxp":p}, ...]<br>
  }<br>
}<br>
