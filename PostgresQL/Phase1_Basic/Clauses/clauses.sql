-- select clause
SELECT * FROM employees;

SELECT emp_id, emp_name FROM employees;

-- where clause
-- operator sign
SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_salary >= 50000;

-- between
SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_salary BETWEEN 50000 AND 70000;

-- in
SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_dept IN ('HR', 'IT');

-- Like
SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_name LIKE '%a%';

SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_name LIKE '_a%';

SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
WHERE emp_dept LIKE '__';


-- Order by
SELECT emp_id, emp_name, emp_salary, emp_dept FROM employees
ORDER BY emp_name DESC;

-- offset-> Skips first 6 rows, returns next 3
SELECT * FROM employees
LIMIT 3 OFFSET 6;

-- limit
SELECT * FROM employees
LIMIT 5;


-- group by
SELECT emp_dept, SUM(emp_salary)
FROM employees
GROUP BY emp_dept;


-- Having clause
-- * WHERE → filters rows
-- * HAVING → filters groups
SELECT emp_dept, SUM(emp_salary)
FROM employees
GROUP BY emp_dept
HAVING SUM(emp_salary) > 50000;












