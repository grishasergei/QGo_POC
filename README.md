# Configuring your development environment
For Python 3
```commandline
pip3 install --upgrade tensorflow
pip3 install keras
pip3 install flask gevent requests pillow
```
 
# Start the API service

```commandline
python run_keras_server.py
```

The service will be running on `http://127.0.0.1:5000/`. You can specify a custom port number:

```commandline
python run_keras_server.py -p 8080
```

# Make predictions using cURL
From a directory where `image.jpg` is located:

```commandline
curl -X POST -F image=@image.jpg 'http://localhost:5000/predict'
```

Example response:
```json
{
    "success": true,
    "total_count": 36,
    "image": [{
            "width": 1280
        },
        {
            "height": 720
        }
    ],
    "patch": [{
            "width": 256
        },
        {
            "height": 256
        }
    ],
    "patches": [{
            "count": 1,
            "patch": 0
        },
        {
            "count": 5,
            "patch": 1
        },
        {
            "count": 8,
            "patch": 2
        },
        {
            "count": 1,
            "patch": 3
        },
        {
            "count": 2,
            "patch": 4
        },
        {
            "count": 1,
            "patch": 5
        },
        {
            "count": 1,
            "patch": 6
        },
        {
            "count": 6,
            "patch": 7
        },
        {
            "count": 6,
            "patch": 8
        },
        {
            "count": 1,
            "patch": 9
        },
        {
            "count": 1,
            "patch": 10
        },
        {
            "count": 1,
            "patch": 11
        },
        {
            "count": 0,
            "patch": 12
        },
        {
            "count": 1,
            "patch": 13
        },
        {
            "count": 1,
            "patch": 14
        }
    ]
}
```
