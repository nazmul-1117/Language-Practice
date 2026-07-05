-- Constrains and datatypes
CREATE TABLE IF NOT EXISTS employees(
	emp_id SERIAL PRIMARY KEY,
	emp_name VARCHAR(50)  NOT NULL,
	emp_email VARCHAR(50) NOT NULL UNIQUE,
	emp_dept VARCHAR(12) NOT NULL,
	emp_salary DECIMAL(8, 2) DEFAULT 30000,
	hire_date DATE NOT NULL DEFAULT CURRENT_DATE
);

-- insert daata
INSERT INTO employees (emp_name, emp_email, emp_dept, emp_salary, hire_date)
VALUES
('Rahim Uddin', 'rahim1@gmail.com', 'IT', 50000, '2024-01-10'),
('Karim Ali', 'karim2@gmail.com', 'HR', 42000, '2023-12-05'),
('Sadia Islam', 'sadia3@gmail.com', 'Finance', 55000, '2024-02-15'),
('Nusrat Jahan', 'nusrat4@gmail.com', 'IT', 60000, '2024-03-01'),
('Tanvir Hasan', 'tanvir5@gmail.com', 'Marketing', 38000, '2023-11-20'),
('Arif Hossain', 'arif6@gmail.com', 'IT', 72000, '2024-01-25'),
('Mehedi Hasan', 'mehedi7@gmail.com', 'Sales', 45000, '2023-10-10'),
('Fariha Noor', 'fariha8@gmail.com', 'HR', 40000, '2024-02-01'),
('Imran Khan', 'imran9@gmail.com', 'Finance', 65000, '2023-09-18'),
('Rakib Hasan', 'rakib10@gmail.com', 'IT', 58000, '2024-04-12');


-- nextvalue
SELECT currval('employees_emp_id_seq') from employees;
SELECT setval('employees_emp_id_seq', 100) from employees;

-- new value insert
INSERT INTO employees (emp_name, emp_email, emp_dept, emp_salary)
VALUES
('Sakib Hasan', 'sakib10@gmail.com', 'HR', 75000);


















