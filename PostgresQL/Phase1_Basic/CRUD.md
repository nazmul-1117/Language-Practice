# 🐘 PostgreSQL CRUD Operations (pgAdmin + SQL Shell)

This document explains how to perform **CRUD (Create, Read, Update, Delete)** operations in PostgreSQL using:

* 🖥️ pgAdmin 4 (Graphical Interface)
* 💻 psql (SQL Shell / Command Line)

---

# 📌 1. Create Database & Table

## 🖥️ Using pgAdmin 4

### Steps:

1. Open pgAdmin 4
2. Connect to your server
3. Right-click → **Databases → Create → Database**
4. Enter database name → Save

### Create Table:

* Go to your database → Schemas → Tables → Create → Table
* Add columns using GUI

OR use Query Tool:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(150) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 💻 Using psql (SQL Shell)

### Login to PostgreSQL:

```bash
psql -U postgres
```

### Create Database:

```sql
CREATE DATABASE mydb;
```

### Connect to Database:

```sql
\c mydb
```

### Create Table:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(150) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Check Data
```sql
\d users
```

---

# 📌 2. INSERT (Create Data)

## 🖥️ pgAdmin 4

* Right-click table → View/Edit Data → All Rows
* Click “+” button → Add values → Save

OR use Query Tool:

```sql
INSERT INTO students(
	id, name, email
)VALUES
(100, 'nazmul', 'nazmul@example.com'),
(101, 'fuad', 'fuad@example.com'),
(102, 'sojib', 'sojib@example.com'),
(103, 'montree', 'montree@example.com'),
(104, 'senapoti', 'senapoti@example.com'),
(105, 'kobi', 'kobi@example.com');
```

---

## 💻 psql

```sql
INSERT INTO students(
	id, name, email
)VALUES
(100, 'nazmul', 'nazmul@example.com'),
(101, 'fuad', 'fuad@example.com'),
(102, 'sojib', 'sojib@example.com'),
(103, 'montree', 'montree@example.com'),
(104, 'senapoti', 'senapoti@example.com'),
(105, 'kobi', 'kobi@example.com');
```

---

# 📌 3. SELECT (Read Data)

## 🖥️ pgAdmin 4

* Right-click table → View/Edit Data → All Rows

OR Query Tool:

```sql
SELECT * FROM students;
```

### Filter Data:

```sql
SELECT * FROM students WHERE id = 101;
```

---

## 💻 psql

```sql
SELECT * FROM students;
```

Pretty output:

```bash
\x
SELECT * FROM students;
```

---

# 📌 4. UPDATE (Modify Data)

## 🖥️ pgAdmin 4

* Open table → Edit row → Save changes

OR Query Tool:

```sql
UPDATE students
SET
	name = 'kobimoshay',
	email = 'kobimoshay@example.com'
WHERE
	id = 105;
```

---

## 💻 psql

```sql
UPDATE students
SET
	name = 'kobimoshay',
	email = 'kobimoshay@example.com'
WHERE
	id = 105;
```

---

# 📌 5. DELETE (Remove Data)

## 🖥️ pgAdmin 4

* Open table → Select row → Delete → Save

OR Query Tool:

```sql
DELETE FROM students
WHERE
	id = 105;
```

---

## 💻 psql

```sql
DELETE FROM students
WHERE
	id = 105;
```

⚠️ Without WHERE = deletes ALL data:

```sql
DELETE FROM students;
```

---

# 📌 6. Useful psql Commands (SQL Shell)

```bash
\l        # List databases
\c dbname # Connect to database
\dt       # List tables
\d table  # Describe table structure
\q        # Quit psql
```

---

# 📌 7. pgAdmin vs psql Summary

| Feature   | pgAdmin 4         | psql (SQL Shell) |
| --------- | ----------------- | ---------------- |
| Interface | GUI               | Command Line     |
| Speed     | Slower            | Faster           |
| Best for  | Beginners         | Developers       |
| Usage     | Visual operations | SQL mastery      |

---

# 🚀 Final Notes

* pgAdmin helps visualize database structure
* psql builds real-world SQL skills (important for interviews)
* Both use the same SQL commands internally

---