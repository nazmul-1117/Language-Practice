-- Employees per Department
SELECT emp_dept, COUNT(*) AS total_employees
FROM employees
GROUP BY emp_dept;

 -- Average Salary per Department
SELECT emp_dept, AVG(emp_salary) AS avg_salary
FROM employees
GROUP BY emp_dept;

-- Total Salary per Department
SELECT emp_dept, SUM(emp_salary) AS total_salary
FROM employees
GROUP BY emp_dept;


-- Min & Max Salary per Department
SELECT emp_dept,
       MIN(emp_salary) AS min_salary,
       MAX(emp_salary) AS max_salary
FROM employees
GROUP BY emp_dept;

-- Departments with more than 2 employees
SELECT emp_dept, COUNT(*) AS total_employees
FROM employees
GROUP BY emp_dept
HAVING COUNT(*) > 2;

-- Departments with total salary > 200000
SELECT emp_dept, SUM(emp_salary) AS total_salary
FROM employees
GROUP BY emp_dept
HAVING SUM(emp_salary) > 200000;
































