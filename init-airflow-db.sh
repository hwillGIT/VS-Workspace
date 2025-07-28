#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;
    GRANT CREATE, CONNECT, TEMPORARY ON DATABASE airflow_db TO airflow;
    ALTER USER airflow WITH SUPERUSER;
EOSQL 