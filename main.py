'''
Dans ce projet vous aurez à mettre en production un modèle de machine learning de votre choix.

Le déploiement du modèle devra se faire automatique via un script simple. 

Il faudra que le modèle puisse gérer une certaine charge. Vous pourrez soit faire du batch processing / une API ou du streaming. 

Il faudra que le système soit capable de : 

soit de se réentraîner tout seul régulièrement grâce à des données nouvellement labelisée
soit de faire remonter des alertes si il existe des risques que le modèle ne marche plus (distributional shift)
Bonus : le modèle sera packagé dans un conteneur docker
'''

# Imports
import uvicorn
from app.api import app
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
#prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "")


# Run server
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
