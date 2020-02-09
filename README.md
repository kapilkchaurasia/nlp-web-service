  
### NLP Web Service to host an ML Model to predict a sentence is relevant or otherwise.  

The container trains a simple [text classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and hosts it for prediction as a web service written in [Sanic Framework](https://sanic.readthedocs.io/en/latest/). The data for model training is included in the project.     
    
---    
 ### Pre-requisites    
    
System should have [docker engine](https://docs.docker.com/install/) installed.    
>**Note**: I developed and tested this on MacOS-mojave.    
 ---    
 ### Hosting the web service    
    
Build the docker image     
```bash 
docker build -t nlp-web-service:v1 .
``` 

Check the image     
```bash 
docker images  
```       
  
Run the container    
```bash
docker run -d -p 8080:8080 --name=nlp-web-service nlp-web-service:v1
```  
  
Check whether the container is up     
```bash 
docker ps  
```   
    
    
>When we run the container two scripts are initiated: 
>1. `train.py` which trains the model to be hosted. 
>2. `prediction.py` which hosts the model as a web service.    
 ---    
 ### API Usage 
 The web services includes the [sanic-openapi](https://github.com/huge-success/sanic-openapi) integration. Thus, we can directly use the swagger portal from web browser to use the API. To open the swagger portal go to your browser and enter `http://localhost:8080/swagger/`. This will open the swagger portal only if the service is hosted properly.    

To check whether service is up: 
```bash   
curl --location --request GET 'localhost:8080/'       
```

To predict the label of the text:    
```bash     
curl --location --request POST 'localhost:8080/v1/predict' \
--header 'Content-Type: application/json' \
--data-raw '{    
	"text": "japan marks 70th anniversary of  hiroshima atomic bombing (from"
}'    
```
---    
 ### Logs checking 
 To check the the web service logs we need to get inside the running container. To do so execute the following command:    
```bash
docker exec -it nlp-web-service bash
``` 
Now we are inside the container.    
    
The logs are available in the `logs` folder in the files `nlp-web-service.log` and `nlp-web-service.err`.    
        
    
---    
 ### Stopping the web service 
 Run the following command to stop the container:    
```bash
docker stop nlp-web-service
```

---
 ### TODO
Read input data from `sqlite` database instead of file.


