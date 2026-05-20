# Local MySQL (Docker)

## Why `DB_SCHEMA` must match `DB_NAME` (MySQL)

`db.qualified_table_name()` builds `` `schema`.`table` ``. In MySQL, *schema* is the database name. If `DB_NAME=parley_iq` but `DB_SCHEMA=public`, queries target the wrong database.

## Quick start

1. **Create `docker/.env`** from the example (passwords are local-only):

   ```bash
   cp docker/.env.example docker/.env
   ```

2. **Start MySQL**

   ```bash
   docker compose up -d
   ```

   Wait until `docker compose ps` shows MySQL healthy.

3. **Point the app at the container** — merge `docker/app.env.example` into your root `.env` (at minimum `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_SCHEMA`).  
   From the host, use:

   - `DB_HOST=127.0.0.1`
   - `DB_PORT=3306` (or the port you set in `MYSQL_PUBLISH_PORT` in `docker/.env`)

4. **Seed dummy conversations** (random **1000–3000** sessions, each **≥5** user/assistant pairs; timestamps are recent so `BATCH_CONVERSATION_SINCE_*` filters still match):

   ```bash
   py scripts/seed_mysql_test_data.py
   ```

   Optional: `py scripts/seed_mysql_test_data.py --sessions 1500` (still clamped to 1000–3000).

5. **Run the pipeline**

   ```bash
   py run.py
   ```

## Reset data

```bash
docker compose down -v
docker compose up -d
py scripts/seed_mysql_test_data.py
```

## Note on `docker/.env` and `MYSQL_PUBLISH_PORT`

`MYSQL_PUBLISH_PORT` is read from the **project root** environment when you run `docker compose` (see `docker-compose.yml`). Set it in your shell or in a root `.env` if Compose loads it; the example keeps `3306`.
