# 🐘 PostgreSQL Clauses

This document explains the most commonly used SQL clauses in PostgreSQL with examples.

Clauses are parts of SQL queries that help you **filter, group, sort, limit, and structure results**.

---

# 📌 1. SELECT Clause

Used to fetch data from a table.

```sql
SELECT * FROM users;
```

Select specific columns:

```sql
SELECT name, email FROM users;
```

---

# 📌 2. WHERE Clause

Used to filter rows based on a condition.

```sql
SELECT * FROM users
WHERE age >= 18;
```

### Common operators:

```sql
=    -- equal
!=   -- not equal
>    -- greater than
<    -- less than
>=
<=
IN
BETWEEN
LIKE
```

Example:

```sql
SELECT * FROM users
WHERE name LIKE 'A%';
```

---

# 📌 3. ORDER BY Clause

Used to sort results.

### Ascending (default)

```sql
SELECT * FROM users
ORDER BY name;
```

### Descending

```sql
SELECT * FROM users
ORDER BY id DESC;
```

---

# 📌 4. LIMIT Clause

Used to restrict number of rows.

```sql
SELECT * FROM users
LIMIT 5;
```

---

# 📌 5. OFFSET Clause

Used to skip rows (pagination).

```sql
SELECT * FROM users
LIMIT 5 OFFSET 10;
```

👉 Skips first 10 rows, returns next 5

---

# 📌 6. GROUP BY Clause

Used to group rows with same values.

```sql
SELECT age, COUNT(*)
FROM users
GROUP BY age;
```

---

# 📌 7. HAVING Clause

Used to filter grouped data.

👉 Difference:

* WHERE → filters rows
* HAVING → filters groups

```sql
SELECT age, COUNT(*)
FROM users
GROUP BY age
HAVING COUNT(*) > 1;
```

---

# 📌 8. DISTINCT Clause

Removes duplicate values.

```sql
SELECT DISTINCT age FROM users;
```

---

# 📌 9. JOIN Clause (Most Important)

Used to combine data from multiple tables.

---

## 🔗 INNER JOIN

Returns matching rows only.

```sql
SELECT users.name, orders.amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;
```

---

## 🔗 LEFT JOIN

Returns all left table rows + matching right table rows.

```sql
SELECT users.name, orders.amount
FROM users
LEFT JOIN orders ON users.id = orders.user_id;
```

---

## 🔗 RIGHT JOIN

Returns all right table rows + matching left table rows.

```sql
SELECT users.name, orders.amount
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;
```

---

## 🔗 FULL JOIN

Returns all rows from both tables.

```sql
SELECT users.name, orders.amount
FROM users
FULL JOIN orders ON users.id = orders.user_id;
```

---

## 🔗 SELF JOIN

A table joins with itself.

```sql
SELECT a.name, b.name
FROM employees a
JOIN employees b
ON a.manager_id = b.id;
```

---

# 📌 10. UNION Clause

Combines results from two queries (removes duplicates).

```sql
SELECT name FROM students
UNION
SELECT name FROM teachers;
```

---

# 📌 11. UNION ALL

Keeps duplicates.

```sql
SELECT name FROM students
UNION ALL
SELECT name FROM teachers;
```

---

# 📌 12. EXISTS Clause

Checks if a subquery returns results.

```sql
SELECT name
FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id
);
```

---

# 📌 13. IN Clause

Matches multiple values.

```sql
SELECT * FROM users
WHERE id IN (1, 2, 3);
```

---

# 📌 14. BETWEEN Clause

Selects range of values.

```sql
SELECT * FROM users
WHERE age BETWEEN 18 AND 30;
```

---

# 📌 15. LIKE Clause

Pattern matching.

```sql
SELECT * FROM users
WHERE name LIKE 'A%';   -- starts with A
```

---

# 🚀 Quick Summary

| Clause   | Purpose         |
| -------- | --------------- |
| SELECT   | Fetch data      |
| WHERE    | Filter rows     |
| ORDER BY | Sort data       |
| LIMIT    | Limit rows      |
| OFFSET   | Skip rows       |
| GROUP BY | Group data      |
| HAVING   | Filter groups   |
| JOIN     | Combine tables  |
| UNION    | Merge results   |
| EXISTS   | Check existence |
| IN       | Multiple match  |
| BETWEEN  | Range filter    |
| LIKE     | Pattern match   |

---

# 🧠 Final Note

Clauses are the **foundation of SQL query building**.
Mastering them is essential for:

* Backend development
* Database design
* Technical interviews