SELECT
    c.customer_unique_id,
    COUNT(o.order_id) AS num_orders,
    AVG(op.payment_value) AS avg_payment_per_order,
    julianday(MAX(o.order_purchase_timestamp)) - julianday(MIN(o.order_purchase_timestamp)) AS days_lifespan
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_payments op ON o.order_id = op.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY c.customer_unique_id
ORDER BY avg_payment_per_order DESC
LIMIT 3
