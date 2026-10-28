-- Executed in the same in-memory SQLite connection after 01_order_mart.sql.
-- run_analysis.py checks agreement with independent pandas monthly metrics.
-- julianday measures elapsed time. date() compares calendar dates, not midnight.
WITH eligible AS (
    SELECT *,
           julianday(order_delivered_customer_date)
             - julianday(order_purchase_timestamp) AS delivery_days,
           CASE WHEN date(order_delivered_customer_date)
                        > date(order_estimated_delivery_date)
                THEN 1 ELSE 0 END AS is_late
    FROM order_mart
    WHERE order_status = 'delivered'
      AND order_purchase_timestamp >= '2017-01-01'
      AND order_purchase_timestamp < '2018-09-01'
      AND order_delivered_customer_date IS NOT NULL
      AND order_estimated_delivery_date IS NOT NULL
      AND customer_unique_id IS NOT NULL
      AND item_count > 0
      AND invalid_item_count = 0
      AND julianday(order_delivered_customer_date)
            >= julianday(order_purchase_timestamp)
      AND (order_approved_at IS NULL OR julianday(order_delivered_customer_date)
            >= julianday(order_approved_at))
      AND (order_delivered_carrier_date IS NULL OR julianday(order_delivered_customer_date)
            >= julianday(order_delivered_carrier_date))
)
SELECT substr(order_purchase_timestamp, 1, 7) AS purchase_month,
       COUNT(*) AS delivered_orders,
       AVG(is_late) AS late_share,
       AVG(delivery_days) AS mean_delivery_days
FROM eligible GROUP BY purchase_month ORDER BY purchase_month;

-- Do not interpret reviews/orders as site conversion: no traffic denominator.
-- Analytical review eligibility and exclusions are implemented in src/data.py.
