'''
Copyright (c) 2019 Mikhail Milchenko, Washington University in Saint Louis
Comments: mmilch@wustl.edu

All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the Washington University in Saint Louis nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import dtext as dt
from flask import Flask,jsonify,request, abort
import threading
import time
import traceback
import logging
import os
import sys
import tensorflow as tf

'''
Ping:
curl http://localhost:5000 

Submit a processing task:
curl -i -H "Content-Type: application/json" -X POST -d '{"path": "/my_path/my_dir"}' http://localhost:5000/process

Responce format:
{
  "task": {
    "id": 1,
    "path": "/my_path/my_dir",
    "status": "created","running","error","aborting","aborted","completed"
    
    #When status is "aborted" or "completed":
    "resultSet":[{'infile':f,'outfile':outfil,'text_present':1,"maxp":p}, ...]    
  }
}

List all tasks:
curl -X GET http://localhost:5000/tasks

Get a specific task json by ID:
curl -X GET http://localhost:5000/tasks/<id>

Abort a running task:
curl -X DELETE http://localhost:5000/tasks/<id>

Delete a non-running task from registry:
curl -X DELETE http://localhost:5000/tasks/<id>

'''

app = Flask(__name__)
lock=threading.Lock()

tasks= []
status= {'status':'online'}

'''
List all processing tasks.
'''
@app.route('/tasks', methods=['GET'])
def list_tasks():
    print('LISTED all tasks')
    return jsonify({'tasks':tasks})

    '''
get json on a task with specific ID
'''
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [t for t in tasks if t['id'] == task_id]
    if len(task) == 0: abort(404)
    print('LISTED task '+str(task_id))
    return jsonify({'task': task[0]})

'''    
      For a running task, request to abort it.
      Otherwise, delete the given task from registry.
'''
@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def del_task(task_id):
    task = [t for t in tasks if t['id'] == task_id]
    if len(task) == 0: abort(404)
    task=task[0]
    if task['status'] == 'running' or task['status'] == 'aborting':
        lock.acquire(); task['status']='aborting'; lock.release()
        print('ABORTING task id: {}'.format(task['path'],task['id']))
        return jsonify({'result': False})
    #if the task wasn't running, can remove it from registry.
    lock.acquire(); tasks.remove(task); lock.release()
    print('DELETED task id: {}'.format(task['path'],task['id']))
    return jsonify({'result': True})

'''
    Run processing on a specified dir.
    Example:
    curl -i -H "Content-Type: application/json" -X POST -d '{"path": "/my_path/my_dir"}' http://localhost:5000/process
    Returns json with task parameters.
'''
@app.route('/process',methods=['POST'])
def process_dir():
    if not request.json or not 'path' in request.json: abort(400)
    new_id=1 if len(tasks)==0 else tasks[-1]['id']+1
    task={
        'id': new_id,
        'path': request.json['path'],
        'status': 'created'
    }
    lock.acquire(); tasks.append(task); lock.release()
    print ('SUBMITTED task id={} to process dir {}'.format(new_id,task['path']))
    return jsonify({'task':task}), 201
    
@app.route('/')
def ping():
    return jsonify(status)

'''
Run a processing task
'''
def run_task(task, model, pmin):
    global tasks
    print('STARTING task id={}, processing dir: {}'.format(task['id'],task['path']))
    lock.acquire(); task['status']='running'; lock.release()
    results=dt.run_detection(model,pmin,task['path'],task['path'],True,task)
    lock.acquire()
    task['resultSet']=results
    task['status'] = 'aborted' if task['status'] == 'aborting' else 'completed'
    lock.release()
    print('FINISHED task id={}, processing dir: {}'.format(task['id'],task['path']))
    
'''
Thread monitoring task execution.
periodically checks for new processing tasks and execute them as they arrive.
'''
def task_monitor():
    print('starting task monitor')
    model_file=os.path.dirname(sys.argv[0])+'./models/09.10.2019.on_5M.hd5'
    pmin=0.99    
    try:
        print('loading model')
        model=tf.keras.models.load_model(model_file)
    except Exception as e:
        logging.error(traceback.format_exc())
        lock.acquire()
        print('error: cannot load model from '+ model_file)
        status={'status':'error: cannot load model from '+ model_file}
        lock.release()
        return
    print('model loaded')
    while True:
        #print('Checking for new tasks')
        try:
            for t in tasks:
                if t['status']=='created':
                    run_task(t, model, pmin)
        except Exception as e:
            logging.error(traceback.format_exc())                    
        time.sleep(1.1)        
    
if __name__ == '__main__':
    #task monitor thread   
    threading.Thread(target=task_monitor).start()    
    #REST server thread
    print('starting REST service')
    app.run(debug=False)
