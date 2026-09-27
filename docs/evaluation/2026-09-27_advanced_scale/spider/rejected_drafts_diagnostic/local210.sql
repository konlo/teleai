SELECT h.hub_name, 
       SUM(CASE WHEN o.order_created_month = 2 AND o.order_status = 'finished' THEN 1 ELSE 0 END) AS feb_finished_orders, 
       SUM(CASE WHEN o.order_created_month = 3 AND o.order_status = 'finished' THEN 1 ELSE 0 END) AS mar_finished_orders
FROM orders o
JOIN stores s ON o.store_id = s.store_id
JOIN hubs h ON s.hub_id = h.hub_id
WHERE o.order_created_year = 2026
GROUP BY h.hub_name
HAVING feb_finished_orders > 0 AND mar_finished_orders > 0 AND (mar_finished_orders - feb_finished_orders) * 100.0 / feb_finished_orders > 20
