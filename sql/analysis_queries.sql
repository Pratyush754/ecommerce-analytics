-- Same analyses as the Python pipeline, written in SQL (PostgreSQL syntax).
-- Load data/raw/*.csv into tables: customers, products, transactions, channel_costs.

-- 1. Monthly net revenue and MoM growth
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month,
         SUM(quantity * unit_price * (1 - discount_pct)) AS revenue
  FROM (SELECT DISTINCT * FROM transactions) t
  WHERE status = 'Delivered' GROUP BY 1)
SELECT month, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly ORDER BY month;

-- 2. Category margin
SELECT p.category,
       SUM(t.quantity * t.unit_price * (1 - t.discount_pct)) AS revenue,
       ROUND(100.0 * (SUM(t.quantity * t.unit_price * (1 - t.discount_pct)) - SUM(t.quantity * p.unit_cost))
             / SUM(t.quantity * t.unit_price * (1 - t.discount_pct)), 1) AS margin_pct
FROM transactions t JOIN products p USING (product_id)
WHERE t.status = 'Delivered' GROUP BY p.category ORDER BY revenue DESC;

-- 3. RFM scoring with window functions
WITH base AS (
  SELECT customer_id,
         DATE '2025-01-01' - MAX(order_date) AS recency,
         COUNT(DISTINCT order_id) AS frequency,
         SUM(quantity * unit_price * (1 - discount_pct)) AS monetary
  FROM transactions WHERE status = 'Delivered' GROUP BY customer_id)
SELECT *, NTILE(5) OVER (ORDER BY recency DESC) AS r,
          NTILE(5) OVER (ORDER BY frequency)    AS f,
          NTILE(5) OVER (ORDER BY monetary)     AS m
FROM base;

-- 4. Cohort retention
WITH o AS (SELECT DISTINCT customer_id, DATE_TRUNC('month', order_date) AS m FROM transactions WHERE status='Delivered'),
c AS (SELECT customer_id, MIN(m) AS cohort FROM o GROUP BY 1)
SELECT c.cohort,
       (EXTRACT(YEAR FROM o.m) - EXTRACT(YEAR FROM c.cohort)) * 12 + EXTRACT(MONTH FROM o.m) - EXTRACT(MONTH FROM c.cohort) AS months_since,
       COUNT(DISTINCT o.customer_id) AS customers
FROM o JOIN c USING (customer_id) GROUP BY 1, 2 ORDER BY 1, 2;

-- 5. Channel LTV : CAC
WITH cp AS (
  SELECT t.customer_id, SUM(t.quantity * (t.unit_price * (1 - t.discount_pct) - p.unit_cost)) AS profit
  FROM transactions t JOIN products p USING (product_id) WHERE t.status = 'Delivered' GROUP BY 1)
SELECT c.channel, ROUND(AVG(COALESCE(cp.profit, 0)), 2) AS ltv, k.cac,
       ROUND(AVG(COALESCE(cp.profit, 0)) / k.cac, 2) AS ltv_cac
FROM customers c LEFT JOIN cp USING (customer_id) JOIN channel_costs k USING (channel)
GROUP BY c.channel, k.cac ORDER BY ltv_cac DESC;
