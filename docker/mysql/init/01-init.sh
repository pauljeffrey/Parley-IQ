#!/bin/bash
# Creates `${DB_SCHEMA}` database (defaults to public) for qualified db names from .env + user_consultation.
set -eu
ROOT_PW="${MYSQL_ROOT_PASSWORD:-}"
SCH="${DB_SCHEMA:-public}"
mysql -uroot -p"${ROOT_PW}" <<EOSQL
CREATE DATABASE IF NOT EXISTS \`${SCH}\`;
GRANT ALL PRIVILEGES ON \`${SCH}\`.* TO '${MYSQL_USER}'@'%';
GRANT ALL PRIVILEGES ON \`${MYSQL_DATABASE}\`.* TO '${MYSQL_USER}'@'%';
FLUSH PRIVILEGES;
USE \`${SCH}\`;
CREATE TABLE IF NOT EXISTS user_consultation (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  session_id VARCHAR(255) NOT NULL,
  user_message TEXT,
  assistant_message TEXT,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  INDEX idx_session_created (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
EOSQL
