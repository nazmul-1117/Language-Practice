# 📊 GROUP BY & HAVING (PostgreSQL) – Using Your Employees Table

---

# 📌 1. GROUP BY Basics

### 🔹 Employees per Department

```sql
SELECT emp_dept, COUNT(*) AS total_employees
FROM employees
GROUP BY emp_dept;
```

👉 This groups employees by department and counts them.

---

### 🔹 Average Salary per Department

```sql
SELECT emp_dept, AVG(emp_salary) AS avg_salary
FROM employees
GROUP BY emp_dept;
```

---

### 🔹 Total Salary per Department

```sql
SELECT emp_dept, SUM(emp_salary) AS total_salary
FROM employees
GROUP BY emp_dept;
```

---

### 🔹 Min & Max Salary per Department

```sql
SELECT emp_dept,
       MIN(emp_salary) AS min_salary,
       MAX(emp_salary) AS max_salary
FROM employees
GROUP BY emp_dept;
```

---

# 📌 2. HAVING Clause (Filter Groups)

👉 Remember:

* WHERE = filters rows before grouping
* HAVING = filters after grouping

---

# 🔹 Departments with more than 2 employees

```sql
SELECT emp_dept, COUNT(*) AS total_employees
FROM employees
GROUP BY emp_dept
HAVING COUNT(*) > 2;
```

---

# 🔹 Departments with total salary > 200000

```sql
SELECT emp_dept, SUM(emp_salary) AS total_salary
FROM employees
GROUP BY emp_dept
HAVING SUM(emp_salary) > 200000;
```

---

# 🔹 Departments with average salary > 50000

```sql
SELECT emp_dept, AVG(emp_salary) AS avg_salary
FROM employees
GROUP BY emp_dept
HAVING AVG(emp_salary) > 50000;
```

---

# 🔹 Departments where max salary is above 60000

```sql
SELECT emp_dept, MAX(emp_salary) AS max_salary
FROM employees
GROUP BY emp_dept
HAVING MAX(emp_salary) > 60000;
```

---

# 📌 3. GROUP BY + ORDER BY (Common Pattern)

### Departments sorted by highest average salary

```sql
SELECT emp_dept, AVG(emp_salary) AS avg_salary
FROM employees
GROUP BY emp_dept
ORDER BY avg_salary DESC;
```

---

# 📌 4. Real Interview-Style Query (Very Important)

### 🔥 Find departments where:

* More than 2 employees exist
* And average salary > 50000

```sql
SELECT emp_dept,
       COUNT(*) AS total_employees,
       AVG(emp_salary) AS avg_salary
FROM employees
GROUP BY emp_dept
HAVING COUNT(*) > 2
   AND AVG(emp_salary) > 50000;
```

---

# 🧠 Quick Concept Summary

| Clause   | Purpose                     |
| -------- | --------------------------- |
| GROUP BY | Group rows                  |
| HAVING   | Filter grouped results      |
| WHERE    | Filter rows before grouping |

---

# 🚀 Key Interview Insight

Most SQL interview problems using your table will follow this pattern:

```sql
SELECT column, AGG_FUNCTION()
FROM table
GROUP BY column
HAVING condition_on_aggregate;
```