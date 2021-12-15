# MLOPS : Modèle de machine learning mis en production

## Description
Mise en production d'un Vision Transformer.
Le endpoint `predict` prend en paramètre l'url d'une image et retourne la classe prédite.
Les queries sont sauvegardés et les classes prédites sont ré-entrainé à l'aide de l'API Unsplash.
Packagé avec Docker.

## Usage
```
docker build -t ml-embedded .
docker run -p 8888:8888 ml-embedded
```
