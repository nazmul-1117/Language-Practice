# 🐘 PostgreSQL Data Types, Constraints & Sequences

This document covers three important PostgreSQL fundamentals:

* 📦 Data Types
* 🔒 Constraints
* 🔢 Sequence functions (`nextval`, `currval`, `setval`)

---

# 📌 1. Data Types in PostgreSQL

PostgreSQL provides rich and flexible data types compared to MySQL.

## 🔤 Common Data Types

### 🧾 Text & String Types

```sql
VARCHAR(n)   -- Variable length with limit
CHAR(n)      -- Fixed length
TEXT         -- Unlimited text
```

---

### 🔢 Numeric Types

```sql
INTEGER / INT
BIGINT
SMALLINT
SERIAL       -- Auto-increment integer (old style)
BIGSERIAL
DECIMAL(p,s) -- Exact precision
NUMERIC(p,s) -- Exact precision
REAL         -- Floating point
```

---

### 📅 Date & Time Types

```sql
DATE         -- Only date
TIME         -- Only time
TIMESTAMP    -- Date + time
TIMESTAMPTZ  -- Timestamp with timezone ⭐ recommended
INTERVAL     -- Time difference
```

---

### 🔘 Boolean Type

```sql
BOOLEAN  -- TRUE / FALSE
```

---

### 📦 Special PostgreSQL Types

```sql
UUID        -- Unique identifier
JSON        -- JSON data
JSONB       -- Binary JSON (faster, recommended)
ARRAY       -- Array values
```

Example:

```sql
CREATE TABLE demo (
    id SERIAL,
    name TEXT,
    tags TEXT[],
    metadata JSONB
);
```

---

# 📌 2. Constraints in PostgreSQL

Constraints are rules applied to table columns to ensure data integrity.

---

## 🔒 PRIMARY KEY

Uniquely identifies each row.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT
);
```

---

## 🔒 NOT NULL

Prevents NULL values.

```sql
name TEXT NOT NULL
```

---

## 🔒 UNIQUE

Ensures no duplicate values.

```sql
email TEXT UNIQUE
```

---

## 🔒 CHECK

Validates condition.

```sql
age INT CHECK (age >= 18)
```

---

## 🔒 FOREIGN KEY

Creates relationship between tables.

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

## 🔒 DEFAULT

Sets default value.

```sql
status TEXT DEFAULT 'active'
```

---

## 🧠 Full Example

```sql
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INT CHECK (age >= 18),
    status TEXT DEFAULT 'active'
);
```

---

# 📌 3. Sequences in PostgreSQL (VERY IMPORTANT)

PostgreSQL uses **sequences** for auto-increment behavior.

This is what powers:

```sql
SERIAL
BIGSERIAL
IDENTITY
```

---

# 🔢 3.1 nextval()

Returns the next value from a sequence.

```sql
SELECT nextval('students_id_seq');
```

👉 Increases sequence automatically

---

# 🔢 3.2 currval()

Returns the current value of the sequence in this session.

```sql
SELECT currval('students_id_seq');
```

⚠️ Important:

* `nextval()` must be called first in the same session
* Otherwise error occurs

---

# 🔢 3.3 setval()

Manually sets sequence value.

```sql
SELECT setval('students_id_seq', 100);
```

---

### 🔧 setval() with is_called option

```sql
SELECT setval('students_id_seq', 100, true);
```

Meaning:

* `true` → next `nextval()` will be 101
* `false` → next `nextval()` will return 100

---

# 🧪 Example Workflow

```sql
-- Create table
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name TEXT
);

-- Insert data
INSERT INTO students (name) VALUES ('Alice');

-- Check sequence
SELECT currval('students_id_seq');

-- Move sequence forward manually
SELECT setval('students_id_seq', 500);

-- Next value
SELECT nextval('students_id_seq');
```

---

# ⚡ SERIAL vs IDENTITY (Modern Way)

### SERIAL (Old style)

```sql
id SERIAL PRIMARY KEY
```

### IDENTITY (Recommended)

```sql
id INT GENERATED ALWAYS AS IDENTITY
```

---

# 🚀 Summary

* Data types define **what kind of data you store**
* Constraints define **rules for data integrity**
* Sequences control **auto-increment behavior**
* `nextval`, `currval`, `setval` give **manual control over sequences**