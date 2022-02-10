#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, redirect, request 
import stackfusionAIAassign


app =  Flask(__name__)

@app.route('/')  
def hello(): 
    return render_template('index.html') 
    
@app.route('/', methods = ['POST']) 

def submit_data():
    if request.method== 'POST':
        f = request.files["userfile"] 
        path = "./static/{}".format(f.filename)
        f.save(path)  
        
        
        path = detect.object(path)

      
        f.save(f.filename) 
        
        path = detection.object(f.filename)

        print(path)
        if path:
            output = main.text_reader(path)
            print(output)
            result_dic = {
            'img' : f.filename,
            'text' : output
        }
        

        else:
            result_dic = {
            'img' : f.filename,
            'text' : "NO number plate detected"
            }

    return(render_template("index.html", your_text = result_dic))
    

if __name__== "__main__":

    app.run(debug=True)


# In[ ]:




