#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE DATABASE hospital_readmission;
    GRANT ALL PRIVILEGES ON DATABASE hospital_readmission TO $POSTGRES_USER;
EOSQL

echo "Database 'hospital_readmission' created successfully"