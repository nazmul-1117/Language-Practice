# 📊 PostgreSQL Aggregate Functions

Aggregate functions perform a calculation on a set of rows and return a single result.

They are widely used in:

* Reports
* Analytics
* Dashboards
* Interview SQL questions

---

# 📌 1. COUNT()

Returns total number of rows.

## 🔹 Count all rows

```sql
SELECT COUNT(*) FROM users;
```

## 🔹 Count non-null values

```sql
SELECT COUNT(email) FROM users;
```

## 🔹 Count distinct values

```sql
SELECT COUNT(DISTINCT age) FROM users;
```

---

# 📌 2. SUM()

Returns total sum of numeric column.

```sql
SELECT SUM(salary) FROM employees;
```

### With condition:

```sql
SELECT SUM(salary)
FROM employees
WHERE department = 'IT';
```

---

# 📌 3. AVG()

Returns average value.

```sql
SELECT AVG(salary) FROM employees;
```

### Rounded average:

```sql
SELECT ROUND(AVG(salary), 2) FROM employees;
```

---

# 📌 4. MIN()

Returns smallest value.

```sql
SELECT MIN(salary) FROM employees;
```

### Example:

```sql
SELECT MIN(age) FROM users;
```

---

# 📌 5. MAX()

Returns largest value.

```sql
SELECT MAX(salary) FROM employees;
```

### Example:

```sql
SELECT MAX(age) FROM users;
```

---

# 📌 6. GROUP BY with Aggregate Functions (VERY IMPORTANT)

Used to group rows before applying aggregates.

## 🔹 Example: count users per age

```sql
SELECT age, COUNT(*)
FROM users
GROUP BY age;
```

---

## 🔹 Example: total salary per department

```sql
SELECT department, SUM(salary)
FROM employees
GROUP BY department;
```

---

## 🔹 Example: average salary per department

```sql
SELECT department, AVG(salary)
FROM employees
GROUP BY department;
```

---

# 📌 7. HAVING with Aggregate Functions

Filters grouped results.

👉 WHERE filters rows
👉 HAVING filters groups

---

## 🔹 Example: departments with more than 5 employees

```sql
SELECT department, COUNT(*)
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;
```

---

## 🔹 Example: departments with average salary > 50k

```sql
SELECT department, AVG(salary)
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```

---

# 📌 8. Combined Example (Real-world Query)

```sql
SELECT department,
       COUNT(*) AS total_employees,
       SUM(salary) AS total_salary,
       AVG(salary) AS avg_salary,
       MIN(salary) AS min_salary,
       MAX(salary) AS max_salary
FROM employees
GROUP BY department;
```

---

# 🧠 Quick Summary

| Function | Purpose        |
| -------- | -------------- |
| COUNT()  | Count rows     |
| SUM()    | Total sum      |
| AVG()    | Average value  |
| MIN()    | Smallest value |
| MAX()    | Largest value  |

---

# 🚀 Why Aggregate Functions Matter

They are essential for:

* Backend reporting systems
* Business analytics
* SQL interviews
* Real-world dashboards

---

# 🎯 Interview Tip

Most common interview pattern:

> "Find department-wise salary statistics"

You will almost always use:

```sql
GROUP BY + COUNT/SUM/AVG + HAVING
```