SELECT DATE(o.order_purchase_timestamp) AS sale_date, SUM(oi.price) AS daily_sales
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-08-30'
  AND (p.product_category_name = 'brinquedos' OR p.product_category_name = 'toys' OR p.product_category_name LIKE '%toy%' OR p.product_category_name LIKE '%brinquedo%')
GROUP BY DATE(o.order_purchase_timestamp)
ORDER BY sale_date
