SELECT o.customer_id, c.customer_unique_id, o.order_purchase_timestamp, oi.price, oi.freight_value
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
LIMIT 5
