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

import os
import json
import re
import csv
import numpy as np
import pickle

__all__ = ('ServerParams','XnatIterator')
class ServerParams:
    '''
    Container parameters received from XNAT
    '''
    def __init__(self,server,user, password, project,subject,experiment):
        self.server=server;
        self.user=user;
        self.password=password;
        self.project=project;
        self.subject=subject;
        self.experiment=experiment;
    def __init__(self,fname):
        with open(fname,'r') as f:
            lines=f.read().splitlines()
            self.server=lines[0];
            self.user=lines[1];
            self.password=lines[2];
            self.project=lines[3];
            self.subject=lines[4];
            self.experiment=lines[5];
            
class XnatIterator:
    def __init__(self,sp=None):
        self.sp=sp
        self._subjects=[]
        self._experiments=[]
        self._scans=[]
        self._jsession=[]
    def connect(self,sp=None):    
        if not sp is None:
            self.sp=sp
        
        cmd="curl -o jsession.txt -k -u "+ self.sp.user+":"+self.sp.password+ \
            " "+self.sp.server+"/data/JSESSION"
        print (cmd)
        os.system(cmd)
        with open("jsession.txt") as f:
            self._jsession=f.read()
            print (self._jsession)
        return not self._jsession is None
            
    def _curl_cmd(self,path):        
        cmd="curl -o temp_query.json -k --cookie JSESSIONID=" + self._jsession+' ' \
            +shlex.quote(self.sp.server+"/data/archive/projects/"+self.sp.project+path)
        !rm -f temp_query.json
        os.system(cmd)
    
    def list_subjects(self):
        self._curl_cmd('/subjects?format=json')
        with open ('temp_query.json') as tq:
            try: 
                df=json.loads(tq.read())
            except:
             #   print ('cannot list subjects')
                return []
        #print(df)
        subjs=sorted(df['ResultSet']['Result'], key=lambda k:k['label'])        
        self._subjects=[f['label'] for f in subjs]
        return self._subjects
    def list_experiments(self,subject):
        self._curl_cmd('/subjects/'+subject+"/experiments?xsiType=xnat:mrSessionData&format=json")        
        with open ('temp_query.json') as tq:
            try: 
                df=json.loads(tq.read())
            except: 
                print ('error listing experiments!')
                return []
        #print(df['ResultSet']['Result'])
        exps=sorted(df['ResultSet']['Result'], key=lambda k:k['date'])
        self._experiments=[f['label'] for f in exps]
        return self._experiments
    def list_scans(self,subject,experiment):
        self._curl_cmd('/subjects/'+ subject +'/experiments/' \
            +experiment + "/scans?columns=ID,frames,type,series_description")
        
        with open ('temp_query.json') as sf:
            try: df=json.loads(sf.read())
            except:
                #print ('cannot list scans')
                return []
        self._scans=sorted(df['ResultSet']['Result'], key=lambda k:k['xnat_imagescandata_id'])
        return self._scans
    '''
    list all scans in project, filtered by subject prefix. 
    Display progres in output textarea.
    Save output in speficified json file.
    '''
    def list_scans_all(self,subjects,subject_prefix,json_out_file,output):
        scans=[]
        for s in subjects:
            if not s.startswith(subject_prefix): continue
            experiments=self.list_experiments(s)
            for e in experiments:
                if output: output.value='{}/{}'.format(s,e)
                scans.append(self.list_scans(s,e))
            with open(json_out_file, 'w') as fp:
                json.dump(scans, fp)
        return scans
        