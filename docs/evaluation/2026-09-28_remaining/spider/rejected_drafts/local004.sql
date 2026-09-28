SELECT 
    o.customer_id,
    COUNT(o.order_id) AS number_of_orders,
    AVG(op.payment_value) AS average_payment_per_order,
    CASE 
        WHEN (julianday(MAX(o.order_purchase_timestamp)) - julianday(MIN(o.order_purchase_timestamp))) / 7 < 1.0 
        THEN 1.0 
        ELSE (julianday(MAX(o.order_purchase_timestamp)) - julianday(MIN(o.order_purchase_timestamp))) / 7 
    END AS customer_lifespan_in_weeks
FROM orders o
JOIN order_payments op ON o.order_id = op.order_id
GROUP BY o.customer_id
ORDER BY average_payment_per_order DESC
LIMIT 3
