SELECT c.Country, SUM(i.Total) AS TotalSales
FROM customers c
JOIN invoices i ON c.CustomerId = i.CustomerId
GROUP BY c.Country
HAVING COUNT(c.CustomerId) > 4
