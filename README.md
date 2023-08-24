# FLFHE
Federated learning with Fully Homomorphic Encryption
## How to install :

Install the requirements in the requirements.txt file

## On Windows :
If there is an compiling error during Pyfhel module installation :
- Go to https://visualstudio.microsoft.com/fr/visual-cpp-build-t0ools/ and install lastest C++ compilers
- Install locally the module :
```
git clone --recursive https://github.com/ibarrond/Pyfhel.git
pip install .
```

## Start federated learning :

Run this command to start the server
```
python server.py 
```

Run this command the number of time you want clients 
```
python client.py -- clientID
```

