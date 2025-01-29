# ./scripts/run_tests.sh
#!/bin/bash
docker-compose run --rm app pytest -v "$@"