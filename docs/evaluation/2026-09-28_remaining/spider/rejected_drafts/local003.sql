WITH delivered_orders AS (
  SELECT
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp,
    SUM(oi.price + oi.freight_value) AS order_total
  FROM customers c
  JOIN orders o ON c.customer_id = o.customer_id
  JOIN order_items oi ON o.order_id = oi.order_id
  JOIN order_payments op ON o.order_id = op.order_id
  WHERE o.order_status = 'delivered'
  GROUP BY c.customer_unique_id, o.order_id, o.order_purchase_timestamp
),
rfm_calc AS (
  SELECT
    customer_unique_id,
    COUNT(order_id) AS frequency,
    SUM(order_total) AS monetary,
    MAX(order_purchase_timestamp) AS last_purchase_date
  FROM delivered_orders
  GROUP BY customer_unique_id
),
rfm_scores AS (
  SELECT
    customer_unique_id,
    frequency,
    monetary,
    last_purchase_date,
    (julianday('now') - julianday(last_purchase_date)) AS recency_days,
    CASE
      WHEN (julianday('now') - julianday(last_purchase_date)) <= 30 THEN 1
      WHEN (julianday('now') - julianday(last_purchase_date)) <= 60 THEN 2
      WHEN (julianday('now') - julianday(last_purchase_date)) <= 90 THEN 3
      WHEN (julianday('now') - julianday(last_purchase_date)) <= 180 THEN 4
      ELSE 5
    END AS recency_score,
    CASE
      WHEN frequency <= 1 THEN 5
      WHEN frequency <= 2 THEN 4
      WHEN frequency <= 4 THEN 3
      WHEN frequency <= 7 THEN 2
      ELSE 1
    END AS frequency_score,
    CASE
      WHEN monetary <= 100 THEN 5
      WHEN monetary <= 200 THEN 4
      WHEN monetary <= 500 THEN 3
      WHEN monetary <= 1000 THEN 2
      ELSE 1
    END AS monetary_score
  FROM rfm_calc
)
SELECT
  rs.recency_score,
  rs.frequency_score,
  rs.monetary_score,
  AVG(do.order_total) AS avg_sales_per_order
FROM delivered_orders do
JOIN rfm_scores rs ON do.customer_unique_id = rs.customer_unique_id
GROUP BY rs.recency_score, rs.frequency_score, rs.monetary_score
ORDER BY rs.recency_score, rs.frequency_score, rs.monetary_score;
