# ./scripts/cleanup.sh
#!/bin/bash
docker-compose down -v
docker system prune -f