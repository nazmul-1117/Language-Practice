That prompt:

```
postgres=#
```

means you are now inside the PostgreSQL interactive SQL shell and can start running SQL commands.

### What you can do next

Here are some common first steps:

**1. List all databases**

```sql
\l
```

**2. Connect to a database**

```sql
\c database_name
```

**3. List tables in current database**

```sql
\dt
```

**4. Show current user**

```sql
SELECT current_user;
```

**5. Create a new database**

```sql
CREATE DATABASE mydb;
```

**6. Exit psql**

```sql
\q
```

### All database Query
```sql
SELECT datname FROM pg_database;
```

> Create Databse
```sql
CREATE DATABASE test;
```

> Switch Databse
```sql
\c test
```

> Drop Databse
```sql
DROP DATABASE test;
```

> Create Databse
```sql

```