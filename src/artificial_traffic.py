import json
import boto3
import time
import random

runtime_client = boto3.client("runtime.sagemaker")

class ArtificialTraffic:   
    def __init__(self, endpointName):
        self.endpointName = endpointName
        self.transactionId = 0

    def increment_id(self):
        self.transactionId += 1
    
    def random_gaussian(self, params=[]):
        mu, sigma = params[0], params[1]
        return random.gauss(mu=mu, sigma=sigma)

    def random_bit(self, params=[]):
        return random.getrandbits(1)

    def random_int(self, params=[]):
        a, b = params[0], params[1]
        return random.randint(a=a, b=b)
        
    def generate_artificial_traffic(self, 
                                    applicationName, 
                                    testIndicator, 
                                    payload, 
                                    size, 
                                    config=[]):
        
        for i in range(size):
            ## monotonically increase transaction id  
            self.increment_id()
            
            ## set custom attributes
            custom_attributes = {
                "testIndicator": testIndicator,
                "applicationName": applicationName,
                "transactionId": self.transactionId,            
            }
            
            ## modify target column values to force violations
            if config:
                for i, k in enumerate(config):
                    func = getattr(ArtificialTraffic, k['function_name'])
                    col_name = k['source']
                    payload[col_name] = str(func(self, params=k['params']))
            
            ## invoke endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=self.endpointName,
                ContentType='application/json',
                Body=json.dumps(payload),
                CustomAttributes=json.dumps(custom_attributes)
            )
            time.sleep(0.15)
            if i > 0 and i % 100 == 0 :
                print('Executed {0} inferences.'.format(i))
        