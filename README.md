Usage:
1. Install [Docker](https://www.docker.com/) Community Edition
2. Run project in docker-environment - `docker-compose up --build`, wait when project starts
([see more about docker-compose](https://docs.docker.com/compose/))
3. Server will start at: [http://localhost:8080](http://localhost:8080), you can go there and search for the documents (it may take a while for the sharded mongo to update so you need to wait until main-server and/or crawler will start logging their actions without mistakes).
